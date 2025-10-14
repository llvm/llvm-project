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
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOpenACC.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

static mlir::LogicalResult emitStmtWithResult(CIRGenFunction &cgf,
                                              const Stmt *exprResult,
                                              AggValueSlot slot,
                                              Address *lastValue) {
  // We have to special case labels here. They are statements, but when put
  // at the end of a statement expression, they yield the value of their
  // subexpression. Handle this by walking through all labels we encounter,
  // emitting them before we evaluate the subexpr.
  // Similar issues arise for attributed statements.
  while (!isa<Expr>(exprResult)) {
    if (const auto *ls = dyn_cast<LabelStmt>(exprResult)) {
      if (cgf.emitLabel(*ls->getDecl()).failed())
        return mlir::failure();
      exprResult = ls->getSubStmt();
    } else if (const auto *as = dyn_cast<AttributedStmt>(exprResult)) {
      // FIXME: Update this if we ever have attributes that affect the
      // semantics of an expression.
      exprResult = as->getSubStmt();
    } else {
      llvm_unreachable("Unknown value statement");
    }
  }

  const Expr *e = cast<Expr>(exprResult);
  QualType exprTy = e->getType();
  if (cgf.hasAggregateEvaluationKind(exprTy)) {
    cgf.emitAggExpr(e, slot);
  } else {
    // We can't return an RValue here because there might be cleanups at
    // the end of the StmtExpr.  Because of that, we have to emit the result
    // here into a temporary alloca.
    cgf.emitAnyExprToMem(e, *lastValue, Qualifiers(),
                         /*IsInit*/ false);
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitCompoundStmtWithoutScope(
    const CompoundStmt &s, Address *lastValue, AggValueSlot slot) {
  mlir::LogicalResult result = mlir::success();
  const Stmt *exprResult = s.getStmtExprResult();
  assert((!lastValue || (lastValue && exprResult)) &&
         "If lastValue is not null then the CompoundStmt must have a "
         "StmtExprResult");

  for (const Stmt *curStmt : s.body()) {
    const bool saveResult = lastValue && exprResult == curStmt;
    if (saveResult) {
      if (emitStmtWithResult(*this, exprResult, slot, lastValue).failed())
        result = mlir::failure();
    } else {
      if (emitStmt(curStmt, /*useCurrentScope=*/false).failed())
        result = mlir::failure();
    }
  }
  return result;
}

mlir::LogicalResult CIRGenFunction::emitCompoundStmt(const CompoundStmt &s,
                                                     Address *lastValue,
                                                     AggValueSlot slot) {
  // Add local scope to track new declared variables.
  SymTableScopeTy varScope(symbolTable);
  mlir::Location scopeLoc = getLoc(s.getSourceRange());
  mlir::OpBuilder::InsertPoint scopeInsPt;
  builder.create<cir::ScopeOp>(
      scopeLoc, [&](mlir::OpBuilder &b, mlir::Type &type, mlir::Location loc) {
        scopeInsPt = b.saveInsertionPoint();
      });
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.restoreInsertionPoint(scopeInsPt);
  LexicalScope lexScope(*this, scopeLoc, builder.getInsertionBlock());
  return emitCompoundStmtWithoutScope(s, lastValue, slot);
}

void CIRGenFunction::emitStopPoint(const Stmt *s) {
  assert(!cir::MissingFeatures::generateDebugInfo());
}

// Build CIR for a statement. useCurrentScope should be true if no new scopes
// need to be created when finding a compound statement.
mlir::LogicalResult CIRGenFunction::emitStmt(const Stmt *s,
                                             bool useCurrentScope,
                                             ArrayRef<const Attr *> attr) {
  if (mlir::succeeded(emitSimpleStmt(s, useCurrentScope)))
    return mlir::success();

  switch (s->getStmtClass()) {
  case Stmt::NoStmtClass:
  case Stmt::CXXCatchStmtClass:
  case Stmt::SEHExceptStmtClass:
  case Stmt::SEHFinallyStmtClass:
  case Stmt::MSDependentExistsStmtClass:
    llvm_unreachable("invalid statement class to emit generically");
  case Stmt::BreakStmtClass:
  case Stmt::NullStmtClass:
  case Stmt::CompoundStmtClass:
  case Stmt::ContinueStmtClass:
  case Stmt::DeclStmtClass:
  case Stmt::ReturnStmtClass:
    llvm_unreachable("should have emitted these statements as simple");

#define STMT(Type, Base)
#define ABSTRACT_STMT(Op)
#define EXPR(Type, Base) case Stmt::Type##Class:
#include "clang/AST/StmtNodes.inc"
    {
      assert(builder.getInsertionBlock() &&
             "expression emission must have an insertion point");

      emitIgnoredExpr(cast<Expr>(s));

      // Classic codegen has a check here to see if the emitter created a new
      // block that isn't used (comparing the incoming and outgoing insertion
      // points) and deletes the outgoing block if it's not used. In CIR, we
      // will handle that during the cir.canonicalize pass.
      return mlir::success();
    }
  case Stmt::IfStmtClass:
    return emitIfStmt(cast<IfStmt>(*s));
  case Stmt::SwitchStmtClass:
    return emitSwitchStmt(cast<SwitchStmt>(*s));
  case Stmt::ForStmtClass:
    return emitForStmt(cast<ForStmt>(*s));
  case Stmt::WhileStmtClass:
    return emitWhileStmt(cast<WhileStmt>(*s));
  case Stmt::DoStmtClass:
    return emitDoStmt(cast<DoStmt>(*s));
  case Stmt::CXXTryStmtClass:
    return emitCXXTryStmt(cast<CXXTryStmt>(*s));
  case Stmt::CXXForRangeStmtClass:
    return emitCXXForRangeStmt(cast<CXXForRangeStmt>(*s), attr);
  case Stmt::OpenACCComputeConstructClass:
    return emitOpenACCComputeConstruct(cast<OpenACCComputeConstruct>(*s));
  case Stmt::OpenACCLoopConstructClass:
    return emitOpenACCLoopConstruct(cast<OpenACCLoopConstruct>(*s));
  case Stmt::OpenACCCombinedConstructClass:
    return emitOpenACCCombinedConstruct(cast<OpenACCCombinedConstruct>(*s));
  case Stmt::OpenACCDataConstructClass:
    return emitOpenACCDataConstruct(cast<OpenACCDataConstruct>(*s));
  case Stmt::OpenACCEnterDataConstructClass:
    return emitOpenACCEnterDataConstruct(cast<OpenACCEnterDataConstruct>(*s));
  case Stmt::OpenACCExitDataConstructClass:
    return emitOpenACCExitDataConstruct(cast<OpenACCExitDataConstruct>(*s));
  case Stmt::OpenACCHostDataConstructClass:
    return emitOpenACCHostDataConstruct(cast<OpenACCHostDataConstruct>(*s));
  case Stmt::OpenACCWaitConstructClass:
    return emitOpenACCWaitConstruct(cast<OpenACCWaitConstruct>(*s));
  case Stmt::OpenACCInitConstructClass:
    return emitOpenACCInitConstruct(cast<OpenACCInitConstruct>(*s));
  case Stmt::OpenACCShutdownConstructClass:
    return emitOpenACCShutdownConstruct(cast<OpenACCShutdownConstruct>(*s));
  case Stmt::OpenACCSetConstructClass:
    return emitOpenACCSetConstruct(cast<OpenACCSetConstruct>(*s));
  case Stmt::OpenACCUpdateConstructClass:
    return emitOpenACCUpdateConstruct(cast<OpenACCUpdateConstruct>(*s));
  case Stmt::OpenACCCacheConstructClass:
    return emitOpenACCCacheConstruct(cast<OpenACCCacheConstruct>(*s));
  case Stmt::OpenACCAtomicConstructClass:
    return emitOpenACCAtomicConstruct(cast<OpenACCAtomicConstruct>(*s));
  case Stmt::GCCAsmStmtClass:
  case Stmt::MSAsmStmtClass:
    return emitAsmStmt(cast<AsmStmt>(*s));
  case Stmt::OMPScopeDirectiveClass:
  case Stmt::OMPErrorDirectiveClass:
  case Stmt::LabelStmtClass:
  case Stmt::AttributedStmtClass:
  case Stmt::GotoStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::CaseStmtClass:
  case Stmt::SEHLeaveStmtClass:
  case Stmt::SYCLKernelCallStmtClass:
  case Stmt::CoroutineBodyStmtClass:
    return emitCoroutineBody(cast<CoroutineBodyStmt>(*s));
  case Stmt::CoreturnStmtClass:
  case Stmt::IndirectGotoStmtClass:
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
  case Stmt::OMPFuseDirectiveClass:
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
    return mlir::failure();
  case Stmt::DeclStmtClass:
    return emitDeclStmt(cast<DeclStmt>(*s));
  case Stmt::CompoundStmtClass:
    if (useCurrentScope)
      return emitCompoundStmtWithoutScope(cast<CompoundStmt>(*s));
    return emitCompoundStmt(cast<CompoundStmt>(*s));
  case Stmt::GotoStmtClass:
    return emitGotoStmt(cast<GotoStmt>(*s));
  case Stmt::ContinueStmtClass:
    return emitContinueStmt(cast<ContinueStmt>(*s));

  // NullStmt doesn't need any handling, but we need to say we handled it.
  case Stmt::NullStmtClass:
    break;

  case Stmt::LabelStmtClass:
    return emitLabelStmt(cast<LabelStmt>(*s));
  case Stmt::CaseStmtClass:
  case Stmt::DefaultStmtClass:
    // If we reached here, we must not handling a switch case in the top level.
    return emitSwitchCase(cast<SwitchCase>(*s),
                          /*buildingTopLevelCase=*/false);
    break;

  case Stmt::BreakStmtClass:
    return emitBreakStmt(cast<BreakStmt>(*s));
  case Stmt::ReturnStmtClass:
    return emitReturnStmt(cast<ReturnStmt>(*s));
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitLabelStmt(const clang::LabelStmt &s) {

  if (emitLabel(*s.getDecl()).failed())
    return mlir::failure();

  if (getContext().getLangOpts().EHAsynch && s.isSideEntry())
    getCIRGenModule().errorNYI(s.getSourceRange(), "IsEHa: not implemented.");

  return emitStmt(s.getSubStmt(), /*useCurrentScope*/ true);
}

// Add a terminating yield on a body region if no other terminators are used.
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

mlir::LogicalResult CIRGenFunction::emitIfStmt(const IfStmt &s) {
  mlir::LogicalResult res = mlir::success();
  // The else branch of a consteval if statement is always the only branch
  // that can be runtime evaluated.
  const Stmt *constevalExecuted;
  if (s.isConsteval()) {
    constevalExecuted = s.isNegatedConsteval() ? s.getThen() : s.getElse();
    if (!constevalExecuted) {
      // No runtime code execution required
      return res;
    }
  }

  // C99 6.8.4.1: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  auto ifStmtBuilder = [&]() -> mlir::LogicalResult {
    if (s.isConsteval())
      return emitStmt(constevalExecuted, /*useCurrentScope=*/true);

    if (s.getInit())
      if (emitStmt(s.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

    if (s.getConditionVariable())
      emitDecl(*s.getConditionVariable());

    // If the condition folds to a constant and this is an 'if constexpr',
    // we simplify it early in CIRGen to avoid emitting the full 'if'.
    bool condConstant;
    if (constantFoldsToBool(s.getCond(), condConstant, s.isConstexpr())) {
      if (s.isConstexpr()) {
        // Handle "if constexpr" explicitly here to avoid generating some
        // ill-formed code since in CIR the "if" is no longer simplified
        // in this lambda like in Clang but postponed to other MLIR
        // passes.
        if (const Stmt *executed = condConstant ? s.getThen() : s.getElse())
          return emitStmt(executed, /*useCurrentScope=*/true);
        // There is nothing to execute at runtime.
        // TODO(cir): there is still an empty cir.scope generated by the caller.
        return mlir::success();
      }
    }

    assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
    assert(!cir::MissingFeatures::incrementProfileCounter());
    return emitIfOnBoolExpr(s.getCond(), s.getThen(), s.getElse());
  };

  // TODO: Add a new scoped symbol table.
  // LexicalScope ConditionScope(*this, S.getCond()->getSourceRange());
  // The if scope contains the full source range for IfStmt.
  mlir::Location scopeLoc = getLoc(s.getSourceRange());
  builder.create<cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
        res = ifStmtBuilder();
      });

  return res;
}

mlir::LogicalResult CIRGenFunction::emitDeclStmt(const DeclStmt &s) {
  assert(builder.getInsertionBlock() && "expected valid insertion point");

  for (const Decl *i : s.decls())
    emitDecl(*i, /*evaluateConditionDecl=*/true);

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitReturnStmt(const ReturnStmt &s) {
  mlir::Location loc = getLoc(s.getSourceRange());
  const Expr *rv = s.getRetValue();

  if (getContext().getLangOpts().ElideConstructors && s.getNRVOCandidate() &&
      s.getNRVOCandidate()->isNRVOVariable()) {
    assert(!cir::MissingFeatures::openMP());
    assert(!cir::MissingFeatures::nrvo());
  } else if (!rv) {
    // No return expression. Do nothing.
  } else if (rv->getType()->isVoidType()) {
    // Make sure not to return anything, but evaluate the expression
    // for side effects.
    if (rv) {
      emitAnyExpr(rv);
    }
  } else if (cast<FunctionDecl>(curGD.getDecl())
                 ->getReturnType()
                 ->isReferenceType()) {
    // If this function returns a reference, take the address of the
    // expression rather than the value.
    RValue result = emitReferenceBindingToExpr(rv);
    builder.CIRBaseBuilderTy::createStore(loc, result.getValue(), *fnRetAlloca);
  } else {
    mlir::Value value = nullptr;
    switch (CIRGenFunction::getEvaluationKind(rv->getType())) {
    case cir::TEK_Scalar:
      value = emitScalarExpr(rv);
      if (value) { // Change this to an assert once emitScalarExpr is complete
        builder.CIRBaseBuilderTy::createStore(loc, value, *fnRetAlloca);
      }
      break;
    case cir::TEK_Complex:
      getCIRGenModule().errorNYI(s.getSourceRange(),
                                 "complex function return type");
      break;
    case cir::TEK_Aggregate:
      assert(!cir::MissingFeatures::aggValueSlotGC());
      emitAggExpr(rv, AggValueSlot::forAddr(returnValue, Qualifiers(),
                                            AggValueSlot::IsDestructed,
                                            AggValueSlot::IsNotAliased,
                                            getOverlapForReturnValue()));
      break;
    }
  }

  auto *retBlock = curLexScope->getOrCreateRetBlock(*this, loc);
  // This should emit a branch through the cleanup block if one exists.
  builder.create<cir::BrOp>(loc, retBlock);
  assert(!cir::MissingFeatures::emitBranchThroughCleanup());
  if (ehStack.stable_begin() != currentCleanupStackDepth)
    cgm.errorNYI(s.getSourceRange(), "return with cleanup stack");

  // Insert the new block to continue codegen after branch to ret block.
  builder.createBlock(builder.getBlock()->getParent());

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitGotoStmt(const clang::GotoStmt &s) {
  // FIXME: LLVM codegen inserts emit a stop point here for debug info
  // sake when the insertion point is available, but doesn't do
  // anything special when there isn't. We haven't implemented debug
  // info support just yet, look at this again once we have it.
  assert(!cir::MissingFeatures::generateDebugInfo());

  cir::GotoOp::create(builder, getLoc(s.getSourceRange()),
                      s.getLabel()->getName());

  // A goto marks the end of a block, create a new one for codegen after
  // emitGotoStmt can resume building in that block.
  // Insert the new block to continue codegen after goto.
  builder.createBlock(builder.getBlock()->getParent());

  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitContinueStmt(const clang::ContinueStmt &s) {
  builder.createContinue(getLoc(s.getKwLoc()));

  // Insert the new block to continue codegen after the continue statement.
  builder.createBlock(builder.getBlock()->getParent());

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitLabel(const clang::LabelDecl &d) {
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
    builder.create<cir::BrOp>(getLoc(d.getSourceRange()), labelBlock);
  }

  builder.setInsertionPointToEnd(labelBlock);
  builder.create<cir::LabelOp>(getLoc(d.getSourceRange()), d.getName());
  builder.setInsertionPointToEnd(labelBlock);

  //  FIXME: emit debug info for labels, incrementProfileCounter
  assert(!cir::MissingFeatures::ehstackBranches());
  assert(!cir::MissingFeatures::incrementProfileCounter());
  assert(!cir::MissingFeatures::generateDebugInfo());
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitBreakStmt(const clang::BreakStmt &s) {
  builder.createBreak(getLoc(s.getKwLoc()));

  // Insert the new block to continue codegen after the break statement.
  builder.createBlock(builder.getBlock()->getParent());

  return mlir::success();
}

template <typename T>
mlir::LogicalResult
CIRGenFunction::emitCaseDefaultCascade(const T *stmt, mlir::Type condType,
                                       mlir::ArrayAttr value, CaseOpKind kind,
                                       bool buildingTopLevelCase) {

  assert((isa<CaseStmt, DefaultStmt>(stmt)) &&
         "only case or default stmt go here");

  mlir::LogicalResult result = mlir::success();

  mlir::Location loc = getLoc(stmt->getBeginLoc());

  enum class SubStmtKind { Case, Default, Other };
  SubStmtKind subStmtKind = SubStmtKind::Other;
  const Stmt *sub = stmt->getSubStmt();

  mlir::OpBuilder::InsertPoint insertPoint;
  builder.create<CaseOp>(loc, value, kind, insertPoint);

  {
    mlir::OpBuilder::InsertionGuard guardSwitch(builder);
    builder.restoreInsertionPoint(insertPoint);

    if (isa<DefaultStmt>(sub) && isa<CaseStmt>(stmt)) {
      subStmtKind = SubStmtKind::Default;
      builder.createYield(loc);
    } else if (isa<CaseStmt>(sub) && isa<DefaultStmt, CaseStmt>(stmt)) {
      subStmtKind = SubStmtKind::Case;
      builder.createYield(loc);
    } else {
      result = emitStmt(sub, /*useCurrentScope=*/!isa<CompoundStmt>(sub));
    }

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
  if (subStmtKind == SubStmtKind::Case) {
    result = emitCaseStmt(*cast<CaseStmt>(sub), condType, buildingTopLevelCase);
  } else if (subStmtKind == SubStmtKind::Default) {
    result = emitDefaultStmt(*cast<DefaultStmt>(sub), condType,
                             buildingTopLevelCase);
  } else if (buildingTopLevelCase) {
    // If we're building a top level case, try to restore the insert point to
    // the case we're building, then we can attach more random stmts to the
    // case to make generating `cir.switch` operation to be a simple form.
    builder.restoreInsertionPoint(insertPoint);
  }

  return result;
}

mlir::LogicalResult CIRGenFunction::emitCaseStmt(const CaseStmt &s,
                                                 mlir::Type condType,
                                                 bool buildingTopLevelCase) {
  cir::CaseOpKind kind;
  mlir::ArrayAttr value;
  llvm::APSInt intVal = s.getLHS()->EvaluateKnownConstInt(getContext());

  // If the case statement has an RHS value, it is representing a GNU
  // case range statement, where LHS is the beginning of the range
  // and RHS is the end of the range.
  if (const Expr *rhs = s.getRHS()) {
    llvm::APSInt endVal = rhs->EvaluateKnownConstInt(getContext());
    value = builder.getArrayAttr({cir::IntAttr::get(condType, intVal),
                                  cir::IntAttr::get(condType, endVal)});
    kind = cir::CaseOpKind::Range;
  } else {
    value = builder.getArrayAttr({cir::IntAttr::get(condType, intVal)});
    kind = cir::CaseOpKind::Equal;
  }

  return emitCaseDefaultCascade(&s, condType, value, kind,
                                buildingTopLevelCase);
}

mlir::LogicalResult CIRGenFunction::emitDefaultStmt(const clang::DefaultStmt &s,
                                                    mlir::Type condType,
                                                    bool buildingTopLevelCase) {
  return emitCaseDefaultCascade(&s, condType, builder.getArrayAttr({}),
                                cir::CaseOpKind::Default, buildingTopLevelCase);
}

mlir::LogicalResult CIRGenFunction::emitSwitchCase(const SwitchCase &s,
                                                   bool buildingTopLevelCase) {
  assert(!condTypeStack.empty() &&
         "build switch case without specifying the type of the condition");

  if (s.getStmtClass() == Stmt::CaseStmtClass)
    return emitCaseStmt(cast<CaseStmt>(s), condTypeStack.back(),
                        buildingTopLevelCase);

  if (s.getStmtClass() == Stmt::DefaultStmtClass)
    return emitDefaultStmt(cast<DefaultStmt>(s), condTypeStack.back(),
                           buildingTopLevelCase);

  llvm_unreachable("expect case or default stmt");
}

mlir::LogicalResult
CIRGenFunction::emitCXXForRangeStmt(const CXXForRangeStmt &s,
                                    ArrayRef<const Attr *> forAttrs) {
  cir::ForOp forOp;

  // TODO(cir): pass in array of attributes.
  auto forStmtBuilder = [&]() -> mlir::LogicalResult {
    mlir::LogicalResult loopRes = mlir::success();
    // Evaluate the first pieces before the loop.
    if (s.getInit())
      if (emitStmt(s.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();
    if (emitStmt(s.getRangeStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();
    if (emitStmt(s.getBeginStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();
    if (emitStmt(s.getEndStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    assert(!cir::MissingFeatures::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!cir::MissingFeatures::requiresCleanups());

    forOp = builder.createFor(
        getLoc(s.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!cir::MissingFeatures::createProfileWeightsForLoop());
          assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          mlir::Value condVal = evaluateExprAsBool(s.getCond());
          builder.createCondition(condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // https://en.cppreference.com/w/cpp/language/for
          // In C++ the scope of the init-statement and the scope of
          // statement are one and the same.
          bool useCurrentScope = true;
          if (emitStmt(s.getLoopVarStmt(), useCurrentScope).failed())
            loopRes = mlir::failure();
          if (emitStmt(s.getBody(), useCurrentScope).failed())
            loopRes = mlir::failure();
          emitStopPoint(&s);
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (s.getInc())
            if (emitStmt(s.getInc(), /*useCurrentScope=*/true).failed())
              loopRes = mlir::failure();
          builder.createYield(loc);
        });
    return loopRes;
  };

  mlir::LogicalResult res = mlir::success();
  mlir::Location scopeLoc = getLoc(s.getSourceRange());
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

  terminateBody(builder, forOp.getBody(), getLoc(s.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitForStmt(const ForStmt &s) {
  cir::ForOp forOp;

  // TODO: pass in an array of attributes.
  auto forStmtBuilder = [&]() -> mlir::LogicalResult {
    mlir::LogicalResult loopRes = mlir::success();
    // Evaluate the first part before the loop.
    if (s.getInit())
      if (emitStmt(s.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();
    assert(!cir::MissingFeatures::loopInfoStack());
    // In the classic codegen, if there are any cleanups between here and the
    // loop-exit scope, a block is created to stage the loop exit. We probably
    // already do the right thing because of ScopeOp, but we need more testing
    // to be sure we handle all cases.
    assert(!cir::MissingFeatures::requiresCleanups());

    forOp = builder.createFor(
        getLoc(s.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!cir::MissingFeatures::createProfileWeightsForLoop());
          assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          mlir::Value condVal;
          if (s.getCond()) {
            // If the for statement has a condition scope,
            // emit the local variable declaration.
            if (s.getConditionVariable())
              emitDecl(*s.getConditionVariable());
            // C99 6.8.5p2/p4: The first substatement is executed if the
            // expression compares unequal to 0. The condition must be a
            // scalar type.
            condVal = evaluateExprAsBool(s.getCond());
          } else {
            condVal = b.create<cir::ConstantOp>(loc, builder.getTrueAttr());
          }
          builder.createCondition(condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // The scope of the for loop body is nested within the scope of the
          // for loop's init-statement and condition.
          if (emitStmt(s.getBody(), /*useCurrentScope=*/false).failed())
            loopRes = mlir::failure();
          emitStopPoint(&s);
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (s.getInc())
            if (emitStmt(s.getInc(), /*useCurrentScope=*/true).failed())
              loopRes = mlir::failure();
          builder.createYield(loc);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(s.getSourceRange());
  builder.create<cir::ScopeOp>(scopeLoc, /*scopeBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Location loc) {
                                 LexicalScope lexScope{
                                     *this, loc, builder.getInsertionBlock()};
                                 res = forStmtBuilder();
                               });

  if (res.failed())
    return res;

  terminateBody(builder, forOp.getBody(), getLoc(s.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitDoStmt(const DoStmt &s) {
  cir::DoWhileOp doWhileOp;

  // TODO: pass in array of attributes.
  auto doStmtBuilder = [&]() -> mlir::LogicalResult {
    mlir::LogicalResult loopRes = mlir::success();
    assert(!cir::MissingFeatures::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!cir::MissingFeatures::requiresCleanups());

    doWhileOp = builder.createDoWhile(
        getLoc(s.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!cir::MissingFeatures::createProfileWeightsForLoop());
          assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          // C99 6.8.5p2/p4: The first substatement is executed if the
          // expression compares unequal to 0. The condition must be a
          // scalar type.
          mlir::Value condVal = evaluateExprAsBool(s.getCond());
          builder.createCondition(condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // The scope of the do-while loop body is a nested scope.
          if (emitStmt(s.getBody(), /*useCurrentScope=*/false).failed())
            loopRes = mlir::failure();
          emitStopPoint(&s);
        });
    return loopRes;
  };

  mlir::LogicalResult res = mlir::success();
  mlir::Location scopeLoc = getLoc(s.getSourceRange());
  builder.create<cir::ScopeOp>(scopeLoc, /*scopeBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Location loc) {
                                 LexicalScope lexScope{
                                     *this, loc, builder.getInsertionBlock()};
                                 res = doStmtBuilder();
                               });

  if (res.failed())
    return res;

  terminateBody(builder, doWhileOp.getBody(), getLoc(s.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitWhileStmt(const WhileStmt &s) {
  cir::WhileOp whileOp;

  // TODO: pass in array of attributes.
  auto whileStmtBuilder = [&]() -> mlir::LogicalResult {
    mlir::LogicalResult loopRes = mlir::success();
    assert(!cir::MissingFeatures::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!cir::MissingFeatures::requiresCleanups());

    whileOp = builder.createWhile(
        getLoc(s.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!cir::MissingFeatures::createProfileWeightsForLoop());
          assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          mlir::Value condVal;
          // If the for statement has a condition scope,
          // emit the local variable declaration.
          if (s.getConditionVariable())
            emitDecl(*s.getConditionVariable());
          // C99 6.8.5p2/p4: The first substatement is executed if the
          // expression compares unequal to 0. The condition must be a
          // scalar type.
          condVal = evaluateExprAsBool(s.getCond());
          builder.createCondition(condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // The scope of the while loop body is a nested scope.
          if (emitStmt(s.getBody(), /*useCurrentScope=*/false).failed())
            loopRes = mlir::failure();
          emitStopPoint(&s);
        });
    return loopRes;
  };

  mlir::LogicalResult res = mlir::success();
  mlir::Location scopeLoc = getLoc(s.getSourceRange());
  builder.create<cir::ScopeOp>(scopeLoc, /*scopeBuilder=*/
                               [&](mlir::OpBuilder &b, mlir::Location loc) {
                                 LexicalScope lexScope{
                                     *this, loc, builder.getInsertionBlock()};
                                 res = whileStmtBuilder();
                               });

  if (res.failed())
    return res;

  terminateBody(builder, whileOp.getBody(), getLoc(s.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitSwitchBody(const Stmt *s) {
  // It is rare but legal if the switch body is not a compound stmt. e.g.,
  //
  //  switch(a)
  //    while(...) {
  //      case1
  //      ...
  //      case2
  //      ...
  //    }
  if (!isa<CompoundStmt>(s))
    return emitStmt(s, /*useCurrentScope=*/true);

  auto *compoundStmt = cast<CompoundStmt>(s);

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

mlir::LogicalResult CIRGenFunction::emitSwitchStmt(const clang::SwitchStmt &s) {
  // TODO: LLVM codegen does some early optimization to fold the condition and
  // only emit live cases. CIR should use MLIR to achieve similar things,
  // nothing to be done here.
  // if (ConstantFoldsToSimpleInteger(S.getCond(), ConstantCondValue))...
  assert(!cir::MissingFeatures::constantFoldSwitchStatement());

  SwitchOp swop;
  auto switchStmtBuilder = [&]() -> mlir::LogicalResult {
    if (s.getInit())
      if (emitStmt(s.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

    if (s.getConditionVariable())
      emitDecl(*s.getConditionVariable(), /*evaluateConditionDecl=*/true);

    mlir::Value condV = emitScalarExpr(s.getCond());

    // TODO: PGO and likelihood (e.g. PGO.haveRegionCounts())
    assert(!cir::MissingFeatures::pgoUse());
    assert(!cir::MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
    // TODO: if the switch has a condition wrapped by __builtin_unpredictable?
    assert(!cir::MissingFeatures::insertBuiltinUnpredictable());

    mlir::LogicalResult res = mlir::success();
    swop = builder.create<SwitchOp>(
        getLoc(s.getBeginLoc()), condV,
        /*switchBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::OperationState &os) {
          curLexScope->setAsSwitch();

          condTypeStack.push_back(condV.getType());

          res = emitSwitchBody(s.getBody());

          condTypeStack.pop_back();
        });

    return res;
  };

  // The switch scope contains the full source range for SwitchStmt.
  mlir::Location scopeLoc = getLoc(s.getSourceRange());
  mlir::LogicalResult res = mlir::success();
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

void CIRGenFunction::emitReturnOfRValue(mlir::Location loc, RValue rv,
                                        QualType ty) {
  if (rv.isScalar()) {
    builder.createStore(loc, rv.getValue(), returnValue);
  } else if (rv.isAggregate()) {
    LValue dest = makeAddrLValue(returnValue, ty);
    LValue src = makeAddrLValue(rv.getAggregateAddress(), ty);
    emitAggregateCopy(dest, src, ty, getOverlapForReturnValue());
  } else {
    cgm.errorNYI(loc, "emitReturnOfRValue: complex return type");
  }
  mlir::Block *retBlock = curLexScope->getOrCreateRetBlock(*this, loc);
  assert(!cir::MissingFeatures::emitBranchThroughCleanup());
  builder.create<cir::BrOp>(loc, retBlock);
  if (ehStack.stable_begin() != currentCleanupStackDepth)
    cgm.errorNYI(loc, "return with cleanup stack");
}
