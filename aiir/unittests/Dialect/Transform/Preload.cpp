//===- Preload.cpp - Test AiirOptMain parameterization ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Transform/DebugExtension/DebugExtension.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/IR/Utils.h"
#include "aiir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "aiir/IR/AsmState.h"
#include "aiir/IR/DialectRegistry.h"
#include "aiir/IR/Verifier.h"
#include "aiir/Parser/Parser.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Support/FileUtilities.h"
#include "aiir/Support/TypeID.h"
#include "aiir/Tools/aiir-opt/AiirOptMain.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace aiir;

namespace aiir {
namespace test {
std::unique_ptr<Pass> createTestTransformDialectInterpreterPass();
} // namespace test
} // namespace aiir

const static llvm::StringLiteral library = R"AIIR(
module attributes {transform.with_named_sequence} {
  transform.named_sequence private @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "from external symbol" : !transform.any_op
    transform.yield
  }
})AIIR";

const static llvm::StringLiteral input = R"AIIR(
module attributes {transform.with_named_sequence} {
  transform.named_sequence private @__transform_main(%arg0: !transform.any_op {transform.readonly})

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    include @__transform_main failures(propagate) (%arg0) : (!transform.any_op) -> ()
  }
})AIIR";

TEST(Preload, ContextPreloadConstructedLibrary) {
  registerPassManagerCLOptions();

  AIIRContext context;
  auto *dialect = context.getOrLoadDialect<transform::TransformDialect>();
  DialectRegistry registry;
  aiir::transform::registerDebugExtension(registry);
  registry.applyExtensions(&context);
  ParserConfig parserConfig(&context);

  OwningOpRef<ModuleOp> inputModule =
      parseSourceString<ModuleOp>(input, parserConfig, "<input>");
  EXPECT_TRUE(inputModule) << "failed to parse input module";

  OwningOpRef<ModuleOp> transformLibrary =
      parseSourceString<ModuleOp>(library, parserConfig, "<transform-library>");
  EXPECT_TRUE(transformLibrary) << "failed to parse transform module";
  LogicalResult diag =
      dialect->loadIntoLibraryModule(std::move(transformLibrary));
  EXPECT_TRUE(succeeded(diag));

  ModuleOp retrievedTransformLibrary =
      transform::detail::getPreloadedTransformModule(&context);
  EXPECT_TRUE(retrievedTransformLibrary)
      << "failed to retrieve transform module";

  OwningOpRef<Operation *> clonedTransformModule(
      retrievedTransformLibrary->clone());

  LogicalResult res = transform::detail::mergeSymbolsInto(
      inputModule->getOperation(), std::move(clonedTransformModule));
  EXPECT_TRUE(succeeded(res)) << "failed to define declared symbols";

  transform::TransformOpInterface entryPoint =
      transform::detail::findTransformEntryPoint(inputModule->getOperation(),
                                                 retrievedTransformLibrary);
  EXPECT_TRUE(entryPoint) << "failed to find entry point";

  transform::TransformOptions options;
  res = transform::applyTransformNamedSequence(
      inputModule->getOperation(), entryPoint, retrievedTransformLibrary,
      options);
  EXPECT_TRUE(succeeded(res)) << "failed to apply named sequence";
}
