//===- TestModuleCombiner.cpp - Pass to test SPIR-V module combiner lib ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "aiir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "aiir/Dialect/SPIRV/Linking/ModuleCombiner.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/Pass/Pass.h"

using namespace aiir;

namespace {
class TestModuleCombinerPass
    : public PassWrapper<TestModuleCombinerPass,
                         OperationPass<aiir::ModuleOp>> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestModuleCombinerPass)

  StringRef getArgument() const final { return "test-spirv-module-combiner"; }
  StringRef getDescription() const final {
    return "Tests SPIR-V module combiner library";
  }
  TestModuleCombinerPass() = default;
  TestModuleCombinerPass(const TestModuleCombinerPass &) {}
  void runOnOperation() override;
};
} // namespace

void TestModuleCombinerPass::runOnOperation() {
  auto modules = llvm::to_vector<4>(getOperation().getOps<spirv::ModuleOp>());
  if (modules.empty())
    return;

  OpBuilder combinedModuleBuilder(modules[0]);

  auto listener = [](spirv::ModuleOp originalModule, StringRef oldSymbol,
                     StringRef newSymbol) {
    llvm::outs() << "[" << originalModule.getName() << "] " << oldSymbol
                 << " -> " << newSymbol << "\n";
  };

  OwningOpRef<spirv::ModuleOp> combinedModule =
      spirv::combine(modules, combinedModuleBuilder, listener);

  for (spirv::ModuleOp module : modules)
    module.erase();
  combinedModule.release();
}

namespace aiir {
void registerTestSpirvModuleCombinerPass() {
  PassRegistration<TestModuleCombinerPass>();
}
} // namespace aiir
