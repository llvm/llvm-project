//===- LibOpt.cpp - Optimize CIR raised C/C++ library idioms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes C/C++ standard library idioms in Clang IR.
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

using cir::CIRBaseBuilderTy;
using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LIBOPT
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct LibOptPass : public impl::LibOptBase<LibOptPass> {
  LibOptPass() = default;
  void runOnOperation() override;

  class PassOptions {
    enum : unsigned {
      None = 0,
      RemarkTransforms = 1,
      RemarkAll = 1 << 1,
    };

  public:
    void parseOption(const llvm::StringRef remark) {
      value |= StringSwitch<unsigned>(remark)
                   .Case("transforms", RemarkTransforms)
                   .Case("all", RemarkAll)
                   .Default(None);
    }

    void parseOptions(LibOptPass &pass) {
      // To be implemented
    }

    bool emitRemarkAll() { return value & RemarkAll; }
    bool emitRemarkTransforms() {
      return emitRemarkAll() || value & RemarkTransforms;
    }

  private:
    unsigned value = None;
  };

  PassOptions passOptions;

  // For now the AST Context is optional for this pass.
  clang::ASTContext *astCtx;
  void setASTContext(clang::ASTContext *c) { astCtx = c; }

  /// Tracks current module.
  ModuleOp theModule;
};
} // namespace

void LibOptPass::runOnOperation() {
  passOptions.parseOptions(*this);
  auto *op = getOperation();
  if (isa<::mlir::ModuleOp>(op))
    theModule = cast<::mlir::ModuleOp>(op);
}

std::unique_ptr<Pass> mlir::createLibOptPass() {
  return std::make_unique<LibOptPass>();
}

std::unique_ptr<Pass> mlir::createLibOptPass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<LibOptPass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
