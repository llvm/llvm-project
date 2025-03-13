//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"

#include "mlir/IR/Builders.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

void CIRGenFunction::emitCompoundStmtWithoutScope(const CompoundStmt &s) {
  for (auto *curStmt : s.body()) {
    if (emitStmt(curStmt, /*useCurrentScope=*/false).failed())
      getCIRGenModule().errorNYI(curStmt->getSourceRange(), "statement");
  }
}

void CIRGenFunction::emitCompoundStmt(const CompoundStmt &s) {
  mlir::Location scopeLoc = getLoc(s.getSourceRange());
  auto scope = builder.create<cir::ScopeOp>(
      scopeLoc, [&](mlir::OpBuilder &b, mlir::Type &type, mlir::Location loc) {
        emitCompoundStmtWithoutScope(s);
      });

  // This code to insert a cir.yield at the end of the scope is temporary until
  // CIRGenFunction::LexicalScope::cleanup() is upstreamed.
  if (!scope.getRegion().empty()) {
    mlir::Block &lastBlock = scope.getRegion().back();
    if (lastBlock.empty() || !lastBlock.mightHaveTerminator() ||
        !lastBlock.getTerminator()->hasTrait<mlir::OpTrait::IsTerminator>()) {
      builder.setInsertionPointToEnd(&lastBlock);
      builder.create<cir::YieldOp>(getLoc(s.getEndLoc()));
    }
  }
}

// Build CIR for a statement. useCurrentScope should be true if no new scopes
// need to be created when finding a compound statement.
mlir::LogicalResult CIRGenFunction::emitStmt(const Stmt *s,
                                             bool useCurrentScope,
                                             ArrayRef<const Attr *> attr) {
  if (mlir::succeeded(emitSimpleStmt(s, useCurrentScope)))
    return mlir::success();

  switch (s->getStmtClass()) {

#define STMT(Type, Base)
#define ABSTRACT_STMT(Op)
#define EXPR(Type, Base) case Stmt::Type##Class:
#include "clang/AST/StmtNodes.inc"
    {
      // Remember the block we came in on.
      mlir::Block *incoming = builder.getInsertionBlock();
      assert(incoming && "expression emission must have an insertion point");

      emitIgnoredExpr(cast<Expr>(s));

      mlir::Block *outgoing = builder.getInsertionBlock();
      assert(outgoing && "expression emission cleared block!");
      return mlir::success();
    }

  case Stmt::OMPScopeDirectiveClass:
  case Stmt::OMPErrorDirectiveClass:
  case Stmt::NoStmtClass:
  case Stmt::CXXCatchStmtClass:
  case Stmt::SEHExceptStmtClass:
  case Stmt::SEHFinallyStmtClass:
  case Stmt::MSDependentExistsStmtClass:
  case Stmt::NullStmtClass:
  case Stmt::CompoundStmtClass:
  case Stmt::DeclStmtClass:
  case Stmt::LabelStmtClass:
  case Stmt::AttributedStmtClass:
  case Stmt::GotoStmtClass:
  case Stmt::BreakStmtClass:
  case Stmt::ContinueStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::CaseStmtClass:
  case Stmt::SEHLeaveStmtClass:
  case Stmt::SYCLKernelCallStmtClass:
  case Stmt::IfStmtClass:
  case Stmt::SwitchStmtClass:
  case Stmt::ForStmtClass:
  case Stmt::WhileStmtClass:
  case Stmt::DoStmtClass:
  case Stmt::CoroutineBodyStmtClass:
  case Stmt::CoreturnStmtClass:
  case Stmt::CXXTryStmtClass:
  case Stmt::CXXForRangeStmtClass:
  case Stmt::IndirectGotoStmtClass:
  case Stmt::ReturnStmtClass:
  case Stmt::GCCAsmStmtClass:
  case Stmt::MSAsmStmtClass:
  case Stmt::OMPParallelDirectiveClass:
  case Stmt::OMPTaskwaitDirectiveClass:
  case Stmt::OMPTaskyieldDirectiveClass:
  case Stmt::OMPBarrierDirectiveClass:
  case Stmt::CapturedStmtClass:
  case Stmt::ObjCAtTryStmtClass:
  case Stmt::ObjCAtThrowStmtClass:
  case Stmt::ObjCAtSynchronizedStmtClass:
  case Stmt::ObjCForCollectionStmtClass:
  case Stmt::ObjCAutoreleasePoolStmtClass:
  case Stmt::SEHTryStmtClass:
  case Stmt::OMPMetaDirectiveClass:
  case Stmt::OMPCanonicalLoopClass:
  case Stmt::OMPSimdDirectiveClass:
  case Stmt::OMPTileDirectiveClass:
  case Stmt::OMPUnrollDirectiveClass:
  case Stmt::OMPForDirectiveClass:
  case Stmt::OMPForSimdDirectiveClass:
  case Stmt::OMPSectionsDirectiveClass:
  case Stmt::OMPSectionDirectiveClass:
  case Stmt::OMPSingleDirectiveClass:
  case Stmt::OMPMasterDirectiveClass:
  case Stmt::OMPCriticalDirectiveClass:
  case Stmt::OMPParallelForDirectiveClass:
  case Stmt::OMPParallelForSimdDirectiveClass:
  case Stmt::OMPParallelMasterDirectiveClass:
  case Stmt::OMPParallelSectionsDirectiveClass:
  case Stmt::OMPTaskDirectiveClass:
  case Stmt::OMPTaskgroupDirectiveClass:
  case Stmt::OMPFlushDirectiveClass:
  case Stmt::OMPDepobjDirectiveClass:
  case Stmt::OMPScanDirectiveClass:
  case Stmt::OMPOrderedDirectiveClass:
  case Stmt::OMPAtomicDirectiveClass:
  case Stmt::OMPTargetDirectiveClass:
  case Stmt::OMPTeamsDirectiveClass:
  case Stmt::OMPCancellationPointDirectiveClass:
  case Stmt::OMPCancelDirectiveClass:
  case Stmt::OMPTargetDataDirectiveClass:
  case Stmt::OMPTargetEnterDataDirectiveClass:
  case Stmt::OMPTargetExitDataDirectiveClass:
  case Stmt::OMPTargetParallelDirectiveClass:
  case Stmt::OMPTargetParallelForDirectiveClass:
  case Stmt::OMPTaskLoopDirectiveClass:
  case Stmt::OMPTaskLoopSimdDirectiveClass:
  case Stmt::OMPMaskedTaskLoopDirectiveClass:
  case Stmt::OMPMaskedTaskLoopSimdDirectiveClass:
  case Stmt::OMPMasterTaskLoopDirectiveClass:
  case Stmt::OMPMasterTaskLoopSimdDirectiveClass:
  case Stmt::OMPParallelGenericLoopDirectiveClass:
  case Stmt::OMPParallelMaskedDirectiveClass:
  case Stmt::OMPParallelMaskedTaskLoopDirectiveClass:
  case Stmt::OMPParallelMaskedTaskLoopSimdDirectiveClass:
  case Stmt::OMPParallelMasterTaskLoopDirectiveClass:
  case Stmt::OMPParallelMasterTaskLoopSimdDirectiveClass:
  case Stmt::OMPDistributeDirectiveClass:
  case Stmt::OMPDistributeParallelForDirectiveClass:
  case Stmt::OMPDistributeParallelForSimdDirectiveClass:
  case Stmt::OMPDistributeSimdDirectiveClass:
  case Stmt::OMPTargetParallelGenericLoopDirectiveClass:
  case Stmt::OMPTargetParallelForSimdDirectiveClass:
  case Stmt::OMPTargetSimdDirectiveClass:
  case Stmt::OMPTargetTeamsGenericLoopDirectiveClass:
  case Stmt::OMPTargetUpdateDirectiveClass:
  case Stmt::OMPTeamsDistributeDirectiveClass:
  case Stmt::OMPTeamsDistributeSimdDirectiveClass:
  case Stmt::OMPTeamsDistributeParallelForSimdDirectiveClass:
  case Stmt::OMPTeamsDistributeParallelForDirectiveClass:
  case Stmt::OMPTeamsGenericLoopDirectiveClass:
  case Stmt::OMPTargetTeamsDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeParallelForDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeParallelForSimdDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeSimdDirectiveClass:
  case Stmt::OMPInteropDirectiveClass:
  case Stmt::OMPDispatchDirectiveClass:
  case Stmt::OMPGenericLoopDirectiveClass:
  case Stmt::OMPReverseDirectiveClass:
  case Stmt::OMPInterchangeDirectiveClass:
  case Stmt::OMPAssumeDirectiveClass:
  case Stmt::OMPMaskedDirectiveClass:
  case Stmt::OMPStripeDirectiveClass:
  case Stmt::OpenACCComputeConstructClass:
  case Stmt::OpenACCLoopConstructClass:
  case Stmt::OpenACCCombinedConstructClass:
  case Stmt::OpenACCDataConstructClass:
  case Stmt::OpenACCEnterDataConstructClass:
  case Stmt::OpenACCExitDataConstructClass:
  case Stmt::OpenACCHostDataConstructClass:
  case Stmt::OpenACCWaitConstructClass:
  case Stmt::OpenACCInitConstructClass:
  case Stmt::OpenACCShutdownConstructClass:
  case Stmt::OpenACCSetConstructClass:
  case Stmt::OpenACCUpdateConstructClass:
  case Stmt::OpenACCCacheConstructClass:
  case Stmt::OpenACCAtomicConstructClass:
  case Stmt::ObjCAtCatchStmtClass:
  case Stmt::ObjCAtFinallyStmtClass:
    cgm.errorNYI(s->getSourceRange(),
                 std::string("emitStmt: ") + s->getStmtClassName());
    return mlir::failure();
  }

  llvm_unreachable("Unexpected statement class");
}

mlir::LogicalResult CIRGenFunction::emitSimpleStmt(const Stmt *s,
                                                   bool useCurrentScope) {
  switch (s->getStmtClass()) {
  default:
    // Only compound and return statements are supported right now.
    return mlir::failure();
  case Stmt::DeclStmtClass:
    return emitDeclStmt(cast<DeclStmt>(*s));
  case Stmt::CompoundStmtClass:
    if (useCurrentScope)
      emitCompoundStmtWithoutScope(cast<CompoundStmt>(*s));
    else
      emitCompoundStmt(cast<CompoundStmt>(*s));
    break;
  case Stmt::ReturnStmtClass:
    return emitReturnStmt(cast<ReturnStmt>(*s));
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitDeclStmt(const DeclStmt &s) {
  assert(builder.getInsertionBlock() && "expected valid insertion point");

  for (const Decl *I : s.decls())
    emitDecl(*I);

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitReturnStmt(const ReturnStmt &s) {
  mlir::Location loc = getLoc(s.getSourceRange());
  const Expr *rv = s.getRetValue();

  if (getContext().getLangOpts().ElideConstructors && s.getNRVOCandidate() &&
      s.getNRVOCandidate()->isNRVOVariable()) {
    getCIRGenModule().errorNYI(s.getSourceRange(),
                               "named return value optimization");
  } else if (!rv) {
    // No return expression. Do nothing.
    // TODO(CIR): In the future when function returns are fully implemented,
    // this section will do nothing.  But for now a ReturnOp is necessary.
    builder.create<ReturnOp>(loc);
  } else if (rv->getType()->isVoidType()) {
    // Make sure not to return anything, but evaluate the expression
    // for side effects.
    if (rv) {
      emitAnyExpr(rv);
    }
  } else if (fnRetTy->isReferenceType()) {
    getCIRGenModule().errorNYI(s.getSourceRange(),
                               "function return type that is a reference");
  } else {
    mlir::Value value = nullptr;
    switch (CIRGenFunction::getEvaluationKind(rv->getType())) {
    case cir::TEK_Scalar:
      value = emitScalarExpr(rv);
      if (value) { // Change this to an assert once emitScalarExpr is complete
        builder.create<ReturnOp>(loc, llvm::ArrayRef(value));
      }
      break;
    default:
      getCIRGenModule().errorNYI(s.getSourceRange(),
                                 "non-scalar function return type");
      break;
    }
  }

  return mlir::success();
}
