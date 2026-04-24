//===- IdiomRecognizer.cpp - recognizing and raising idioms to CIR --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for recognizing idioms (such as uses of functions
// and types to the C/C++ standard library) and replacing them with Clang IR
// operators for later optimization.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_IDIOMRECOGNIZER
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct IdiomRecognizerPass
    : public impl::IdiomRecognizerBase<IdiomRecognizerPass> {
  IdiomRecognizerPass() = default;

  void runOnOperation() override;

  void recognizeStandardLibraryCall(CallOp call);

  clang::ASTContext *astCtx;
  void setASTContext(clang::ASTContext *c) { astCtx = c; }

  /// Tracks current module.
  ModuleOp theModule;
};
} // namespace

void IdiomRecognizerPass::recognizeStandardLibraryCall(CallOp call) {
  // To be implemented
}

void IdiomRecognizerPass::runOnOperation() {
  // The AST context will be used to provide additional information such as
  // namespaces and template parameter lists that are lost after lowering to
  // CIR. This information is necessary to recognize many idioms, such as calls
  // to standard library functions.

  // For now, the AST will be required to allow for faster prototyping and
  // exploring of new optimizations. In the future, it may be preferable to
  // make it optional to reduce memory pressure and allow this pass to run
  // on standalone CIR assembly (Possibly generated from non-Clang front ends).

  assert(astCtx && "Missing ASTContext, please construct with the right ctor");
  theModule = getOperation();

  // Process call operations
  theModule->walk([&](CallOp callOp) {
    // Skip indirect calls.
    std::optional<llvm::StringRef> callee = callOp.getCallee();
    if (!callee)
      return;

    recognizeStandardLibraryCall(callOp);
  });
}

std::unique_ptr<Pass> mlir::createIdiomRecognizerPass() {
  return std::make_unique<IdiomRecognizerPass>();
}

std::unique_ptr<Pass>
mlir::createIdiomRecognizerPass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<IdiomRecognizerPass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
