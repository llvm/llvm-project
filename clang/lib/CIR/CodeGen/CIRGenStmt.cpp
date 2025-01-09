//===--- CIRGenStmt.cpp - Emit CIR Code from Statements -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "Address.h"
#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "mlir/IR/Value.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Stmt.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

Address CIRGenFunction::emitCompoundStmtWithoutScope(const CompoundStmt &S,
                                                     bool getLast,
                                                     AggValueSlot slot) {
  const Stmt *ExprResult = S.getStmtExprResult();
  assert((!getLast || (getLast && ExprResult)) &&
         "If getLast is true then the CompoundStmt must have a StmtExprResult");

  Address retAlloca = Address::invalid();

  for (auto *CurStmt : S.body()) {
    if (getLast && ExprResult == CurStmt) {
      while (!isa<Expr>(ExprResult)) {
        if (const auto *LS = dyn_cast<LabelStmt>(ExprResult))
          llvm_unreachable("labels are NYI");
        else if (const auto *AS = dyn_cast<AttributedStmt>(ExprResult))
          llvm_unreachable("statement attributes are NYI");
        else
          llvm_unreachable("Unknown value statement");
      }

      const Expr *E = cast<Expr>(ExprResult);
      QualType exprTy = E->getType();
      if (hasAggregateEvaluationKind(exprTy)) {
        emitAggExpr(E, slot);
      } else {
        // We can't return an RValue here because there might be cleanups at
        // the end of the StmtExpr.  Because of that, we have to emit the result
        // here into a temporary alloca.
        retAlloca = CreateMemTemp(exprTy, getLoc(E->getSourceRange()));
        emitAnyExprToMem(E, retAlloca, Qualifiers(),
                         /*IsInit*/ false);
      }
    } else {
      if (emitStmt(CurStmt, /*useCurrentScope=*/false).failed())
        llvm_unreachable("failed to build statement");
    }
  }

  return retAlloca;
}

Address CIRGenFunction::emitCompoundStmt(const CompoundStmt &S, bool getLast,
                                         AggValueSlot slot) {
  Address retAlloca = Address::invalid();

  // Add local scope to track new declared variables.
  SymTableScopeTy varScope(symbolTable);
  auto scopeLoc = getLoc(S.getSourceRange());
  mlir::OpBuilder::InsertPoint scopeInsPt;
  builder.create<cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Type &type, mlir::Location loc) {
        scopeInsPt = b.saveInsertionPoint();
      });
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.restoreInsertionPoint(scopeInsPt);
    LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
    retAlloca = emitCompoundStmtWithoutScope(S, getLast, slot);
  }

  return retAlloca;
}

void CIRGenFunction::emitStopPoint(const Stmt *S) {
  assert(!cir::MissingFeatures::generateDebugInfo());
}

// Build CIR for a statement. useCurrentScope should be true if no
// new scopes need be created when finding a compound statement.
mlir::LogicalResult CIRGenFunction::emitStmt(const Stmt *S,
                                             bool useCurrentScope,
                                             ArrayRef<const Attr *> Attrs) {
  if (mlir::succeeded(emitSimpleStmt(S, useCurrentScope)))
    return mlir::success();

  if (getContext().getLangOpts().OpenMP &&
      getContext().getLangOpts().OpenMPSimd)
    assert(0 && "not implemented");

  switch (S->getStmtClass()) {
  case Stmt::OMPScopeDirectiveClass:
    llvm_unreachable("NYI");
  case Stmt::OpenACCCombinedConstructClass:
  case Stmt::OpenACCComputeConstructClass:
  case Stmt::OpenACCLoopConstructClass:
  case Stmt::OMPErrorDirectiveClass:
  case Stmt::NoStmtClass:
  case Stmt::CXXCatchStmtClass:
  case Stmt::SEHExceptStmtClass:
  case Stmt::SEHFinallyStmtClass:
  case Stmt::MSDependentExistsStmtClass:
    llvm_unreachable("invalid statement class to emit generically");
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
    llvm_unreachable("should have emitted these statements as simple");

#define STMT(Type, Base)
#define ABSTRACT_STMT(Op)
#define EXPR(Type, Base) case Stmt::Type##Class:
#include "clang/AST/StmtNodes.inc"
    {
      // Remember the block we came in on.
      mlir::Block *incoming = builder.getInsertionBlock();
      assert(incoming && "expression emission must have an insertion point");

      emitIgnoredExpr(cast<Expr>(S));

      mlir::Block *outgoing = builder.getInsertionBlock();
      assert(outgoing && "expression emission cleared block!");

      break;
    }

  case Stmt::IfStmtClass:
    if (emitIfStmt(cast<IfStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::SwitchStmtClass:
    if (emitSwitchStmt(cast<SwitchStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::ForStmtClass:
    if (emitForStmt(cast<ForStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::WhileStmtClass:
    if (emitWhileStmt(cast<WhileStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::DoStmtClass:
    if (emitDoStmt(cast<DoStmt>(*S)).failed())
      return mlir::failure();
    break;

  case Stmt::CoroutineBodyStmtClass:
    return emitCoroutineBody(cast<CoroutineBodyStmt>(*S));
  case Stmt::CoreturnStmtClass:
    return emitCoreturnStmt(cast<CoreturnStmt>(*S));

  case Stmt::CXXTryStmtClass:
    return emitCXXTryStmt(cast<CXXTryStmt>(*S));

  case Stmt::CXXForRangeStmtClass:
    return emitCXXForRangeStmt(cast<CXXForRangeStmt>(*S), Attrs);

  case Stmt::IndirectGotoStmtClass:
  case Stmt::ReturnStmtClass:
  // When implemented, GCCAsmStmtClass should fall-through to MSAsmStmtClass.
  case Stmt::GCCAsmStmtClass:
  case Stmt::MSAsmStmtClass:
    return emitAsmStmt(cast<AsmStmt>(*S));
  // OMP directives:
  case Stmt::OMPParallelDirectiveClass:
    return emitOMPParallelDirective(cast<OMPParallelDirective>(*S));
  case Stmt::OMPTaskwaitDirectiveClass:
    return emitOMPTaskwaitDirective(cast<OMPTaskwaitDirective>(*S));
  case Stmt::OMPTaskyieldDirectiveClass:
    return emitOMPTaskyieldDirective(cast<OMPTaskyieldDirective>(*S));
  case Stmt::OMPBarrierDirectiveClass:
    return emitOMPBarrierDirective(cast<OMPBarrierDirective>(*S));
  // Unsupported AST nodes:
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
  case Stmt::OMPMaskedDirectiveClass: {
    llvm::errs() << "CIR codegen for '" << S->getStmtClassName()
                 << "' not implemented\n";
    assert(0 && "not implemented");
    break;
  }
  case Stmt::ObjCAtCatchStmtClass:
    llvm_unreachable(
        "@catch statements should be handled by EmitObjCAtTryStmt");
  case Stmt::ObjCAtFinallyStmtClass:
    llvm_unreachable(
        "@finally statements should be handled by EmitObjCAtTryStmt");
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitSimpleStmt(const Stmt *S,
                                                   bool useCurrentScope) {
  switch (S->getStmtClass()) {
  default:
    return mlir::failure();
  case Stmt::DeclStmtClass:
    return emitDeclStmt(cast<DeclStmt>(*S));
  case Stmt::CompoundStmtClass:
    useCurrentScope ? emitCompoundStmtWithoutScope(cast<CompoundStmt>(*S))
                    : emitCompoundStmt(cast<CompoundStmt>(*S));
    break;
  case Stmt::ReturnStmtClass:
    return emitReturnStmt(cast<ReturnStmt>(*S));
  case Stmt::GotoStmtClass:
    return emitGotoStmt(cast<GotoStmt>(*S));
  case Stmt::ContinueStmtClass:
    return emitContinueStmt(cast<ContinueStmt>(*S));
  case Stmt::NullStmtClass:
    break;

  case Stmt::LabelStmtClass:
    return emitLabelStmt(cast<LabelStmt>(*S));

  case Stmt::CaseStmtClass:
  case Stmt::DefaultStmtClass:
    // If we reached here, we must not handling a switch case in the top level.
    return emitSwitchCase(cast<SwitchCase>(*S),
                          /*buildingTopLevelCase=*/false);
    break;

  case Stmt::BreakStmtClass:
    return emitBreakStmt(cast<BreakStmt>(*S));

  case Stmt::AttributedStmtClass:
    return emitAttributedStmt(cast<AttributedStmt>(*S));

  case Stmt::SEHLeaveStmtClass:
    llvm::errs() << "CIR codegen for '" << S->getStmtClassName()
                 << "' not implemented\n";
    assert(0 && "not implemented");
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitLabelStmt(const clang::LabelStmt &S) {
  if (emitLabel(S.getDecl()).failed())
    return mlir::failure();

  // IsEHa: not implemented.
  assert(!(getContext().getLangOpts().EHAsynch && S.isSideEntry()));

  return emitStmt(S.getSubStmt(), /* useCurrentScope */ true);
}

mlir::LogicalResult
CIRGenFunction::emitAttributedStmt(const AttributedStmt &S) {
  for (const auto *A : S.getAttrs()) {
    switch (A->getKind()) {
    case attr::NoMerge:
    case attr::NoInline:
    case attr::AlwaysInline:
    case attr::MustTail:
      llvm_unreachable("NIY attributes");
    case attr::CXXAssume: {
      const Expr *assumption = cast<CXXAssumeAttr>(A)->getAssumption();
      if (getLangOpts().CXXAssumptions && builder.getInsertionBlock() &&
          !assumption->HasSideEffects(getContext())) {
        mlir::Value assumptionValue = emitCheckedArgForAssume(assumption);
        builder.create<cir::AssumeOp>(getLoc(S.getSourceRange()),
                                      assumptionValue);
      }
      break;
    }
    default:
      break;
    }
  }

  return emitStmt(S.getSubStmt(), true, S.getAttrs());
}

// Add terminating yield on body regions (loops, ...) in case there are
// not other terminators used.
// FIXME: make terminateCaseRegion use this too.
static void terminateBody(CIRGenBuilderTy &builder, mlir::Region &r,
                          mlir::Location loc) {
  if (r.empty())
    return;

  SmallVector<mlir::Block *, 4> eraseBlocks;
  unsigned numBlocks = r.getBlocks().size();
  for (auto &block : r.getBlocks()) {
    // Already cleanup after return operations, which might create
    // empty blocks if emitted as last stmt.
    if (numBlocks != 1 && block.empty() && block.hasNoPredecessors() &&
        block.hasNoSuccessors())
      eraseBlocks.push_back(&block);

    if (block.empty() ||
        !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::OpBuilder::InsertionGuard guardCase(builder);
      builder.setInsertionPointToEnd(&block);
      builder.createYield(loc);
    }
  }

  for (auto *b : eraseBlocks)
    b->erase();
}

mlir::LogicalResult CIRGenFunction::emitIfStmt(const IfStmt &S) {
  mlir::LogicalResult res = mlir::success();
  // The else branch of a consteval if statement is always the only branch
  // that can be runtime evaluated.
  const Stmt *ConstevalExecuted;
  if (S.isConsteval()) {
    ConstevalExecuted = S.isNegatedConsteval() ? S.getThen() : S.getElse();
    if (!ConstevalExecuted)
      // No runtime code execution required
      return res;
  }

  // C99 6.8.4.1: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  auto ifStmtBuilder = [&]() -> mlir::LogicalResult {
    if (S.isConsteval())
      return emitStmt(ConstevalExecuted, /*useCurrentScope=*/true);

    if (S.getInit())
      if (emitStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

    if (S.getConditionVariable())
      emitDecl(*S.getConditionVariable());

    // During LLVM codegen, if the condition constant folds and can be elided,
    // it tries to avoid emitting the condition and the dead arm of the if/else.
    // TODO(cir): we skip this in CIRGen, but should implement this as part of
    // SSCP or a specific CIR pass.
    bool CondConstant;
    if (ConstantFoldsToSimpleInteger(S.getCond(), CondConstant,
                                     S.isConstexpr())) {
      if (S.isConstexpr()) {
        // Handle "if constexpr" explicitly here to avoid generating some
        // ill-formed code since in CIR the "if" is no longer simplified
        // in this lambda like in Clang but postponed to other MLIR
        // passes.
        if (const Stmt *Executed = CondConstant ? S.getThen() : S.getElse())
          return emitStmt(Executed, /*useCurrentScope=*/true);
        // There is nothing to execute at runtime.
        // TODO(cir): there is still an empty cir.scope generated by the caller.
        return mlir::success();
      }
      assert(!cir::MissingFeatures::constantFoldsToSimpleInteger());
    }

    assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
    assert(!cir::MissingFeatures::incrementProfileCounter());
    return emitIfOnBoolExpr(S.getCond(), S.getThen(), S.getElse());
  };

  // TODO: Add a new scoped symbol table.
  // LexicalScope ConditionScope(*this, S.getCond()->getSourceRange());
  // The if scope contains the full source range for IfStmt.
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        res = ifStmtBuilder();
      });

  return res;
}

mlir::LogicalResult CIRGenFunction::emitDeclStmt(const DeclStmt &S) {
  if (!builder.getInsertionBlock()) {
    CGM.emitError("Seems like this is unreachable code, what should we do?");
    return mlir::failure();
  }

  for (const auto *I : S.decls()) {
    emitDecl(*I);
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitReturnStmt(const ReturnStmt &S) {
  assert(!cir::MissingFeatures::requiresReturnValueCheck());
  assert(!cir::MissingFeatures::isSEHTryScope());

  auto loc = getLoc(S.getSourceRange());

  // Emit the result value, even if unused, to evaluate the side effects.
  const Expr *RV = S.getRetValue();

  // Record the result expression of the return statement. The recorded
  // expression is used to determine whether a block capture's lifetime should
  // end at the end of the full expression as opposed to the end of the scope
  // enclosing the block expression.
  //
  // This permits a small, easily-implemented exception to our over-conservative
  // rules about not jumping to statements following block literals with
  // non-trivial cleanups.
  // TODO(cir): SaveRetExpr
  // SaveRetExprRAII SaveRetExpr(RV, *this);

  RunCleanupsScope cleanupScope(*this);
  bool createNewScope = false;
  if (const auto *EWC = dyn_cast_or_null<ExprWithCleanups>(RV)) {
    RV = EWC->getSubExpr();
    createNewScope = true;
  }

  auto handleReturnVal = [&]() {
    if (getContext().getLangOpts().ElideConstructors && S.getNRVOCandidate() &&
        S.getNRVOCandidate()->isNRVOVariable()) {
      assert(!cir::MissingFeatures::openMP());
      // Apply the named return value optimization for this return statement,
      // which means doing nothing: the appropriate result has already been
      // constructed into the NRVO variable.

      // If there is an NRVO flag for this variable, set it to 1 into indicate
      // that the cleanup code should not destroy the variable.
      if (auto NRVOFlag = NRVOFlags[S.getNRVOCandidate()])
        getBuilder().createFlagStore(loc, true, NRVOFlag);
    } else if (!ReturnValue.isValid() || (RV && RV->getType()->isVoidType())) {
      // Make sure not to return anything, but evaluate the expression
      // for side effects.
      if (RV) {
        emitAnyExpr(RV);
      }
    } else if (!RV) {
      // Do nothing (return value is left uninitialized)
    } else if (FnRetTy->isReferenceType()) {
      // If this function returns a reference, take the address of the
      // expression rather than the value.
      RValue Result = emitReferenceBindingToExpr(RV);
      builder.createStore(loc, Result.getScalarVal(), ReturnValue);
    } else {
      mlir::Value V = nullptr;
      switch (CIRGenFunction::getEvaluationKind(RV->getType())) {
      case cir::TEK_Scalar:
        V = emitScalarExpr(RV);
        builder.CIRBaseBuilderTy::createStore(loc, V, *FnRetAlloca);
        break;
      case cir::TEK_Complex:
        emitComplexExprIntoLValue(RV,
                                  makeAddrLValue(ReturnValue, RV->getType()),
                                  /*isInit*/ true);
        break;
      case cir::TEK_Aggregate:
        emitAggExpr(
            RV, AggValueSlot::forAddr(
                    ReturnValue, Qualifiers(), AggValueSlot::IsDestructed,
                    AggValueSlot::DoesNotNeedGCBarriers,
                    AggValueSlot::IsNotAliased, getOverlapForReturnValue()));
        break;
      }
    }
  };

  if (!createNewScope)
    handleReturnVal();
  else {
    mlir::Location scopeLoc =
        getLoc(RV ? RV->getSourceRange() : S.getSourceRange());
    // First create cir.scope and later emit it's body. Otherwise all CIRGen
    // dispatched by `handleReturnVal()` might needs to manipulate blocks and
    // look into parents, which are all unlinked.
    mlir::OpBuilder::InsertPoint scopeBody;
    builder.create<cir::ScopeOp>(scopeLoc, /*scopeBuilder=*/
                                 [&](mlir::OpBuilder &b, mlir::Location loc) {
                                   scopeBody = b.saveInsertionPoint();
                                 });
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.restoreInsertionPoint(scopeBody);
      CIRGenFunction::LexicalScope lexScope{*this, scopeLoc,
                                            builder.getInsertionBlock()};
      handleReturnVal();
    }
  }

  cleanupScope.ForceCleanup();

  // In CIR we might have returns in different scopes.
  // FIXME(cir): cleanup code is handling actual return emission, the logic
  // should try to match traditional codegen more closely (to the extend which
  // is possible).
  auto *retBlock = currLexScope->getOrCreateRetBlock(*this, loc);
  emitBranchThroughCleanup(loc, returnBlock(retBlock));

  // Insert the new block to continue codegen after branch to ret block.
  builder.createBlock(builder.getBlock()->getParent());
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitGotoStmt(const GotoStmt &S) {
  // FIXME: LLVM codegen inserts emit stop point here for debug info
  // sake when the insertion point is available, but doesn't do
  // anything special when there isn't. We haven't implemented debug
  // info support just yet, look at this again once we have it.
  assert(builder.getInsertionBlock() && "not yet implemented");

  builder.create<cir::GotoOp>(getLoc(S.getSourceRange()),
                              S.getLabel()->getName());

  // A goto marks the end of a block, create a new one for codegen after
  // emitGotoStmt can resume building in that block.
  // Insert the new block to continue codegen after goto.
  builder.createBlock(builder.getBlock()->getParent());

  // What here...
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitLabel(const LabelDecl *D) {
  // Create a new block to tag with a label and add a branch from
  // the current one to it. If the block is empty just call attach it
  // to this label.
  mlir::Block *currBlock = builder.getBlock();
  mlir::Block *labelBlock = currBlock;
  if (!currBlock->empty()) {
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      labelBlock = builder.createBlock(builder.getBlock()->getParent());
    }
    builder.create<BrOp>(getLoc(D->getSourceRange()), labelBlock);
  }

  builder.setInsertionPointToEnd(labelBlock);
  builder.create<cir::LabelOp>(getLoc(D->getSourceRange()), D->getName());
  builder.setInsertionPointToEnd(labelBlock);

  //  FIXME: emit debug info for labels, incrementProfileCounter
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitContinueStmt(const clang::ContinueStmt &S) {
  builder.createContinue(getLoc(S.getContinueLoc()));

  // Insert the new block to continue codegen after the continue statement.
  builder.createBlock(builder.getBlock()->getParent());

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitBreakStmt(const clang::BreakStmt &S) {
  builder.createBreak(getLoc(S.getBreakLoc()));

  // Insert the new block to continue codegen after the break statement.
  builder.createBlock(builder.getBlock()->getParent());

  return mlir::success();
}

const CaseStmt *CIRGenFunction::foldCaseStmt(const clang::CaseStmt &S,
                                             mlir::Type condType,
                                             mlir::ArrayAttr &value,
                                             cir::CaseOpKind &kind) {
  const CaseStmt *caseStmt = &S;
  const CaseStmt *lastCase = &S;
  SmallVector<mlir::Attribute, 4> caseEltValueListAttr;

  // Fold cascading cases whenever possible to simplify codegen a bit.
  while (caseStmt) {
    lastCase = caseStmt;

    auto intVal = caseStmt->getLHS()->EvaluateKnownConstInt(getContext());

    if (auto *rhs = caseStmt->getRHS()) {
      auto endVal = rhs->EvaluateKnownConstInt(getContext());
      SmallVector<mlir::Attribute, 4> rangeCaseAttr = {
          cir::IntAttr::get(condType, intVal),
          cir::IntAttr::get(condType, endVal)};
      value = builder.getArrayAttr(rangeCaseAttr);
      kind = cir::CaseOpKind::Range;

      // We may not be able to fold rangaes. Due to we can't present range case
      // with other trivial cases now.
      return caseStmt;
    }

    caseEltValueListAttr.push_back(cir::IntAttr::get(condType, intVal));

    caseStmt = dyn_cast_or_null<CaseStmt>(caseStmt->getSubStmt());

    // Break early if we found ranges. We can't fold ranges due to the same
    // reason above.
    if (caseStmt && caseStmt->getRHS())
      break;
  }

  if (!caseEltValueListAttr.empty()) {
    value = builder.getArrayAttr(caseEltValueListAttr);
    kind = caseEltValueListAttr.size() > 1 ? cir::CaseOpKind::Anyof
                                           : cir::CaseOpKind::Equal;
  }

  return lastCase;
}

template <typename T>
mlir::LogicalResult
CIRGenFunction::emitCaseDefaultCascade(const T *stmt, mlir::Type condType,
                                       mlir::ArrayAttr value, CaseOpKind kind,
                                       bool buildingTopLevelCase) {

  assert((isa<CaseStmt, DefaultStmt>(stmt)) &&
         "only case or default stmt go here");

  mlir::LogicalResult result = mlir::success();

  auto loc = getLoc(stmt->getBeginLoc());

  enum class SubStmtKind { Case, Default, Other };
  SubStmtKind subStmtKind = SubStmtKind::Other;
  auto *sub = stmt->getSubStmt();

  mlir::OpBuilder::InsertPoint insertPoint;
  builder.create<CaseOp>(loc, value, kind, insertPoint);

  {
    mlir::OpBuilder::InsertionGuard guardSwitch(builder);
    builder.restoreInsertionPoint(insertPoint);

    if (isa<DefaultStmt>(sub) && isa<CaseStmt>(stmt)) {
      subStmtKind = SubStmtKind::Default;
      builder.createYield(loc);
    } else if (isa<CaseStmt>(sub) && isa<DefaultStmt>(stmt)) {
      subStmtKind = SubStmtKind::Case;
      builder.createYield(loc);
    } else
      result = emitStmt(sub, /*useCurrentScope=*/!isa<CompoundStmt>(sub));

    insertPoint = builder.saveInsertionPoint();
  }

  // If the substmt is default stmt or case stmt, try to handle the special case
  // to make it into the simple form. e.g.
  //
  //  swtich () {
  //    case 1:
  //    default:
  //      ...
  //  }
  //
  // we prefer generating
  //
  //  cir.switch() {
  //     cir.case(equal, 1) {
  //        cir.yield
  //     }
  //     cir.case(default) {
  //        ...
  //     }
  //  }
  //
  // than
  //
  //  cir.switch() {
  //     cir.case(equal, 1) {
  //       cir.case(default) {
  //         ...
  //       }
  //     }
  //  }
  //
  // We don't need to revert this if we find the current switch can't be in
  // simple form later since the conversion itself should be harmless.
  if (subStmtKind == SubStmtKind::Case)
    result = emitCaseStmt(*cast<CaseStmt>(sub), condType, buildingTopLevelCase);
  else if (subStmtKind == SubStmtKind::Default)
    result = emitDefaultStmt(*cast<DefaultStmt>(sub), condType,
                             buildingTopLevelCase);
  else if (buildingTopLevelCase)
    // If we're building a top level case, try to restore the insert point to
    // the case we're building, then we can attach more random stmts to the
    // case to make generating `cir.switch` operation to be a simple form.
    builder.restoreInsertionPoint(insertPoint);

  return result;
}

mlir::LogicalResult CIRGenFunction::emitCaseStmt(const CaseStmt &S,
                                                 mlir::Type condType,
                                                 bool buildingTopLevelCase) {
  mlir::ArrayAttr value;
  CaseOpKind kind;
  auto *caseStmt = foldCaseStmt(S, condType, value, kind);
  return emitCaseDefaultCascade(caseStmt, condType, value, kind,
                                buildingTopLevelCase);
}

mlir::LogicalResult CIRGenFunction::emitDefaultStmt(const DefaultStmt &S,
                                                    mlir::Type condType,
                                                    bool buildingTopLevelCase) {
  return emitCaseDefaultCascade(&S, condType, builder.getArrayAttr({}),
                                cir::CaseOpKind::Default, buildingTopLevelCase);
}

mlir::LogicalResult CIRGenFunction::emitSwitchCase(const SwitchCase &S,
                                                   bool buildingTopLevelCase) {
  assert(!condTypeStack.empty() &&
         "build switch case without specifying the type of the condition");

  if (S.getStmtClass() == Stmt::CaseStmtClass)
    return emitCaseStmt(cast<CaseStmt>(S), condTypeStack.back(),
                        buildingTopLevelCase);

  if (S.getStmtClass() == Stmt::DefaultStmtClass)
    return emitDefaultStmt(cast<DefaultStmt>(S), condTypeStack.back(),
                           buildingTopLevelCase);

  llvm_unreachable("expect case or default stmt");
}

mlir::LogicalResult
CIRGenFunction::emitCXXForRangeStmt(const CXXForRangeStmt &S,
                                    ArrayRef<const Attr *> ForAttrs) {
  cir::ForOp forOp;

  // TODO(cir): pass in array of attributes.
  auto forStmtBuilder = [&]() -> mlir::LogicalResult {
    auto loopRes = mlir::success();
    // Evaluate the first pieces before the loop.
    if (S.getInit())
      if (emitStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();
    if (emitStmt(S.getRangeStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();
    if (emitStmt(S.getBeginStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();
    if (emitStmt(S.getEndStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    assert(!cir::MissingFeatures::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!cir::MissingFeatures::requiresCleanups());

    forOp = builder.createFor(
        getLoc(S.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!cir::MissingFeatures::createProfileWeightsForLoop());
          assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          mlir::Value condVal = evaluateExprAsBool(S.getCond());
          builder.createCondition(condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // https://en.cppreference.com/w/cpp/language/for
          // In C++ the scope of the init-statement and the scope of
          // statement are one and the same.
          bool useCurrentScope = true;
          if (emitStmt(S.getLoopVarStmt(), useCurrentScope).failed())
            loopRes = mlir::failure();
          if (emitStmt(S.getBody(), useCurrentScope).failed())
            loopRes = mlir::failure();
          emitStopPoint(&S);
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (S.getInc())
            if (emitStmt(S.getInc(), /*useCurrentScope=*/true).failed())
              loopRes = mlir::failure();
          builder.createYield(loc);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<cir::ScopeOp>(scopeLoc, /*scopeBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Location loc) {
                                 // Create a cleanup scope for the condition
                                 // variable cleanups. Logical equivalent from
                                 // LLVM codegn for LexicalScope
                                 // ConditionScope(*this, S.getSourceRange())...
                                 LexicalScope lexScope{
                                     *this, loc, builder.getInsertionBlock()};
                                 res = forStmtBuilder();
                               });

  if (res.failed())
    return res;

  terminateBody(builder, forOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitForStmt(const ForStmt &S) {
  cir::ForOp forOp;

  // TODO: pass in array of attributes.
  auto forStmtBuilder = [&]() -> mlir::LogicalResult {
    auto loopRes = mlir::success();
    // Evaluate the first part before the loop.
    if (S.getInit())
      if (emitStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();
    assert(!cir::MissingFeatures::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!cir::MissingFeatures::requiresCleanups());

    forOp = builder.createFor(
        getLoc(S.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!cir::MissingFeatures::createProfileWeightsForLoop());
          assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          mlir::Value condVal;
          if (S.getCond()) {
            // If the for statement has a condition scope,
            // emit the local variable declaration.
            if (S.getConditionVariable())
              emitDecl(*S.getConditionVariable());
            // C99 6.8.5p2/p4: The first substatement is executed if the
            // expression compares unequal to 0. The condition must be a
            // scalar type.
            condVal = evaluateExprAsBool(S.getCond());
          } else {
            auto boolTy = cir::BoolType::get(b.getContext());
            condVal = b.create<cir::ConstantOp>(
                loc, boolTy, cir::BoolAttr::get(b.getContext(), boolTy, true));
          }
          builder.createCondition(condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // The scope of the for loop body is nested within the scope of the
          // for loop's init-statement and condition.
          if (emitStmt(S.getBody(), /*useCurrentScope=*/false).failed())
            loopRes = mlir::failure();
          emitStopPoint(&S);
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (S.getInc())
            if (emitStmt(S.getInc(), /*useCurrentScope=*/true).failed())
              loopRes = mlir::failure();
          builder.createYield(loc);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<cir::ScopeOp>(scopeLoc, /*scopeBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Location loc) {
                                 LexicalScope lexScope{
                                     *this, loc, builder.getInsertionBlock()};
                                 res = forStmtBuilder();
                               });

  if (res.failed())
    return res;

  terminateBody(builder, forOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitDoStmt(const DoStmt &S) {
  cir::DoWhileOp doWhileOp;

  // TODO: pass in array of attributes.
  auto doStmtBuilder = [&]() -> mlir::LogicalResult {
    auto loopRes = mlir::success();
    assert(!cir::MissingFeatures::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!cir::MissingFeatures::requiresCleanups());

    doWhileOp = builder.createDoWhile(
        getLoc(S.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!cir::MissingFeatures::createProfileWeightsForLoop());
          assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          // C99 6.8.5p2/p4: The first substatement is executed if the
          // expression compares unequal to 0. The condition must be a
          // scalar type.
          mlir::Value condVal = evaluateExprAsBool(S.getCond());
          builder.createCondition(condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // The scope of the do-while loop body is a nested scope.
          if (emitStmt(S.getBody(), /*useCurrentScope=*/false).failed())
            loopRes = mlir::failure();
          emitStopPoint(&S);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<cir::ScopeOp>(scopeLoc, /*scopeBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Location loc) {
                                 LexicalScope lexScope{
                                     *this, loc, builder.getInsertionBlock()};
                                 res = doStmtBuilder();
                               });

  if (res.failed())
    return res;

  terminateBody(builder, doWhileOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitWhileStmt(const WhileStmt &S) {
  cir::WhileOp whileOp;

  // TODO: pass in array of attributes.
  auto whileStmtBuilder = [&]() -> mlir::LogicalResult {
    auto loopRes = mlir::success();
    assert(!cir::MissingFeatures::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!cir::MissingFeatures::requiresCleanups());

    whileOp = builder.createWhile(
        getLoc(S.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!cir::MissingFeatures::createProfileWeightsForLoop());
          assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          mlir::Value condVal;
          // If the for statement has a condition scope,
          // emit the local variable declaration.
          if (S.getConditionVariable())
            emitDecl(*S.getConditionVariable());
          // C99 6.8.5p2/p4: The first substatement is executed if the
          // expression compares unequal to 0. The condition must be a
          // scalar type.
          condVal = evaluateExprAsBool(S.getCond());
          builder.createCondition(condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // The scope of the while loop body is a nested scope.
          if (emitStmt(S.getBody(), /*useCurrentScope=*/false).failed())
            loopRes = mlir::failure();
          emitStopPoint(&S);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<cir::ScopeOp>(scopeLoc, /*scopeBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Location loc) {
                                 LexicalScope lexScope{
                                     *this, loc, builder.getInsertionBlock()};
                                 res = whileStmtBuilder();
                               });

  if (res.failed())
    return res;

  terminateBody(builder, whileOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitSwitchBody(const Stmt *S) {
  // It is rare but legal if the switch body is not a compound stmt. e.g.,
  //
  //  switch(a)
  //    while(...) {
  //      case1
  //      ...
  //      case2
  //      ...
  //    }
  if (!isa<CompoundStmt>(S))
    return emitStmt(S, /*useCurrentScope=*/!false);

  auto *compoundStmt = cast<CompoundStmt>(S);

  mlir::Block *swtichBlock = builder.getBlock();
  for (auto *c : compoundStmt->body()) {
    if (auto *switchCase = dyn_cast<SwitchCase>(c)) {
      builder.setInsertionPointToEnd(swtichBlock);
      // Reset insert point automatically, so that we can attach following
      // random stmt to the region of previous built case op to try to make
      // the being generated `cir.switch` to be in simple form.
      if (mlir::failed(
              emitSwitchCase(*switchCase, /*buildingTopLevelCase=*/true)))
        return mlir::failure();

      continue;
    }

    // Otherwise, just build the statements in the nearest case region.
    if (mlir::failed(emitStmt(c, /*useCurrentScope=*/!isa<CompoundStmt>(c))))
      return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitSwitchStmt(const SwitchStmt &S) {
  // TODO: LLVM codegen does some early optimization to fold the condition and
  // only emit live cases. CIR should use MLIR to achieve similar things,
  // nothing to be done here.
  // if (ConstantFoldsToSimpleInteger(S.getCond(), ConstantCondValue))...

  SwitchOp swop;
  auto switchStmtBuilder = [&]() -> mlir::LogicalResult {
    if (S.getInit())
      if (emitStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

    if (S.getConditionVariable())
      emitDecl(*S.getConditionVariable());

    mlir::Value condV = emitScalarExpr(S.getCond());

    // TODO: PGO and likelihood (e.g. PGO.haveRegionCounts())
    // TODO: if the switch has a condition wrapped by __builtin_unpredictable?

    auto res = mlir::success();
    swop = builder.create<SwitchOp>(
        getLoc(S.getBeginLoc()), condV,
        /*switchBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::OperationState &os) {
          currLexScope->setAsSwitch();

          condTypeStack.push_back(condV.getType());

          res = emitSwitchBody(S.getBody());

          condTypeStack.pop_back();
        });

    return res;
  };

  // The switch scope contains the full source range for SwitchStmt.
  auto scopeLoc = getLoc(S.getSourceRange());
  auto res = mlir::success();
  builder.create<cir::ScopeOp>(scopeLoc, /*scopeBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Location loc) {
                                 LexicalScope lexScope{
                                     *this, loc, builder.getInsertionBlock()};
                                 res = switchStmtBuilder();
                               });

  llvm::SmallVector<CaseOp> cases;
  swop.collectCases(cases);
  for (auto caseOp : cases)
    terminateBody(builder, caseOp.getCaseRegion(), caseOp.getLoc());
  terminateBody(builder, swop.getBody(), swop.getLoc());

  return res;
}

void CIRGenFunction::emitReturnOfRValue(mlir::Location loc, RValue RV,
                                        QualType Ty) {
  if (RV.isScalar()) {
    builder.createStore(loc, RV.getScalarVal(), ReturnValue);
  } else if (RV.isAggregate()) {
    LValue Dest = makeAddrLValue(ReturnValue, Ty);
    LValue Src = makeAddrLValue(RV.getAggregateAddress(), Ty);
    emitAggregateCopy(Dest, Src, Ty, getOverlapForReturnValue());
  } else {
    llvm_unreachable("NYI");
  }
  auto *retBlock = currLexScope->getOrCreateRetBlock(*this, loc);
  emitBranchThroughCleanup(loc, returnBlock(retBlock));
}
