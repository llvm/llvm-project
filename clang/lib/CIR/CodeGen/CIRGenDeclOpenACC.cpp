//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Decl nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "clang/AST/DeclOpenACC.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {
struct OpenACCDeclareCleanup final : EHScopeStack::Cleanup {
  mlir::acc::DeclareEnterOp enterOp;

  OpenACCDeclareCleanup(mlir::acc::DeclareEnterOp enterOp) : enterOp(enterOp) {}

  void emit(CIRGenFunction &cgf) override {
    mlir::acc::DeclareExitOp::create(cgf.getBuilder(), enterOp.getLoc(),
                                     enterOp, {});

    // TODO(OpenACC): Some clauses require that we add info about them to the
    // DeclareExitOp.  However, we don't have any of those implemented yet, so
    // we should add infrastructure here to do that once we have one
    // implemented.
  }
};
} // namespace

void CIRGenFunction::emitOpenACCDeclare(const OpenACCDeclareDecl &d) {
  mlir::Location exprLoc = cgm.getLoc(d.getBeginLoc());
  auto enterOp = mlir::acc::DeclareEnterOp::create(
      builder, exprLoc, mlir::acc::DeclareTokenType::get(&cgm.getMLIRContext()),
      {});

  emitOpenACCClauses(enterOp, OpenACCDirectiveKind::Declare, d.getBeginLoc(),
                     d.clauses());

  ehStack.pushCleanup<OpenACCDeclareCleanup>(CleanupKind::NormalCleanup,
                                             enterOp);
}

void CIRGenFunction::emitOpenACCRoutine(const OpenACCRoutineDecl &d) {
  getCIRGenModule().errorNYI(d.getSourceRange(), "OpenACC Routine Construct");
}

void CIRGenModule::emitGlobalOpenACCDecl(const OpenACCConstructDecl *d) {
  if (isa<OpenACCRoutineDecl>(d))
    errorNYI(d->getSourceRange(), "OpenACC Routine Construct");
  else if (isa<OpenACCDeclareDecl>(d))
    errorNYI(d->getSourceRange(), "OpenACC Declare Construct");
  else
    llvm_unreachable("unknown OpenACC declaration kind?");
}
