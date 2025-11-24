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
  SourceRange declareRange;
  mlir::acc::DeclareEnterOp enterOp;

  OpenACCDeclareCleanup(SourceRange declareRange,
                        mlir::acc::DeclareEnterOp enterOp)
      : declareRange(declareRange), enterOp(enterOp) {}

  template <typename OutTy, typename InTy>
  void createOutOp(CIRGenFunction &cgf, InTy inOp) {
    if constexpr (std::is_same_v<OutTy, mlir::acc::DeleteOp>) {
      auto outOp =
          OutTy::create(cgf.getBuilder(), inOp.getLoc(), inOp,
                        inOp.getStructured(), inOp.getImplicit(),
                        llvm::Twine(inOp.getNameAttr()), inOp.getBounds());
      outOp.setDataClause(inOp.getDataClause());
      outOp.setModifiers(inOp.getModifiers());
    } else {
      auto outOp =
          OutTy::create(cgf.getBuilder(), inOp.getLoc(), inOp, inOp.getVarPtr(),
                        inOp.getStructured(), inOp.getImplicit(),
                        llvm::Twine(inOp.getNameAttr()), inOp.getBounds());
      outOp.setDataClause(inOp.getDataClause());
      outOp.setModifiers(inOp.getModifiers());
    }
  }

  void emit(CIRGenFunction &cgf) override {
    auto exitOp = mlir::acc::DeclareExitOp::create(
        cgf.getBuilder(), enterOp.getLoc(), enterOp, {});

    // Some data clauses need to be referenced in 'exit', AND need to have an
    // operation after the exit.  Copy these from the enter operation.
    for (mlir::Value val : enterOp.getDataClauseOperands()) {
      if (auto copyin = val.getDefiningOp<mlir::acc::CopyinOp>()) {
        switch (copyin.getDataClause()) {
        default:
          cgf.cgm.errorNYI(declareRange,
                           "OpenACC local declare clause copyin cleanup");
          break;
        case mlir::acc::DataClause::acc_copy:
          createOutOp<mlir::acc::CopyoutOp>(cgf, copyin);
          break;
        case mlir::acc::DataClause::acc_copyin:
          createOutOp<mlir::acc::DeleteOp>(cgf, copyin);
          break;
        }
      } else if (auto create = val.getDefiningOp<mlir::acc::CreateOp>()) {
        switch (create.getDataClause()) {
        default:
          cgf.cgm.errorNYI(declareRange,
                           "OpenACC local declare clause create cleanup");
          break;
        case mlir::acc::DataClause::acc_copyout:
          createOutOp<mlir::acc::CopyoutOp>(cgf, create);
          break;
        case mlir::acc::DataClause::acc_create:
          createOutOp<mlir::acc::DeleteOp>(cgf, create);
          break;
        }
      } else if (val.getDefiningOp<mlir::acc::DeclareLinkOp>()) {
        // Link has no exit clauses, and shouldn't be copied.
        continue;
      } else if (val.getDefiningOp<mlir::acc::DevicePtrOp>()) {
        // DevicePtr has no exit clauses, and shouldn't be copied.
        continue;
      } else {
        cgf.cgm.errorNYI(declareRange, "OpenACC local declare clause cleanup");
        continue;
      }
      exitOp.getDataClauseOperandsMutable().append(val);
    }
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
                                             d.getSourceRange(), enterOp);
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
