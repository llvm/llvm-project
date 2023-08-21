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

#include "CIRGenFunction.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

mlir::LogicalResult
CIRGenFunction::buildCompoundStmtWithoutScope(const CompoundStmt &S) {
  for (auto *CurStmt : S.body())
    if (buildStmt(CurStmt, /*useCurrentScope=*/false).failed())
      return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildCompoundStmt(const CompoundStmt &S) {
  mlir::LogicalResult res = mlir::success();

  auto compoundStmtBuilder = [&]() -> mlir::LogicalResult {
    if (buildCompoundStmtWithoutScope(S).failed())
      return mlir::failure();

    return mlir::success();
  };

  // Add local scope to track new declared variables.
  SymTableScopeTy varScope(symbolTable);
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScopeContext lexScope{loc, builder.getInsertionBlock()};
        LexicalScopeGuard lexScopeGuard{*this, &lexScope};
        res = compoundStmtBuilder();
      });

  return res;
}

void CIRGenFunction::buildStopPoint(const Stmt *S) {
  assert(!UnimplementedFeature::generateDebugInfo());
}

// Build CIR for a statement. useCurrentScope should be true if no
// new scopes need be created when finding a compound statement.
mlir::LogicalResult CIRGenFunction::buildStmt(const Stmt *S,
                                              bool useCurrentScope,
                                              ArrayRef<const Attr *> Attrs) {
  if (mlir::succeeded(buildSimpleStmt(S, useCurrentScope)))
    return mlir::success();

  if (getContext().getLangOpts().OpenMP &&
      getContext().getLangOpts().OpenMPSimd)
    assert(0 && "not implemented");

  switch (S->getStmtClass()) {
  case Stmt::OMPScopeDirectiveClass:
    llvm_unreachable("NYI");
  case Stmt::OpenACCComputeConstructClass:
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

      buildIgnoredExpr(cast<Expr>(S));

      mlir::Block *outgoing = builder.getInsertionBlock();
      assert(outgoing && "expression emission cleared block!");

      // FIXME: Should we mimic LLVM emission here?
      // The expression emitters assume (reasonably!) that the insertion
      // point is always set.  To maintain that, the call-emission code
      // for noreturn functions has to enter a new block with no
      // predecessors.  We want to kill that block and mark the current
      // insertion point unreachable in the common case of a call like
      // "exit();".  Since expression emission doesn't otherwise create
      // blocks with no predecessors, we can just test for that.
      // However, we must be careful not to do this to our incoming
      // block, because *statement* emission does sometimes create
      // reachable blocks which will have no predecessors until later in
      // the function.  This occurs with, e.g., labels that are not
      // reachable by fallthrough.
      if (incoming != outgoing && outgoing->use_empty())
        assert(0 && "not implemented");
      break;
    }

  case Stmt::IfStmtClass:
    if (buildIfStmt(cast<IfStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::SwitchStmtClass:
    if (buildSwitchStmt(cast<SwitchStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::ForStmtClass:
    if (buildForStmt(cast<ForStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::WhileStmtClass:
    if (buildWhileStmt(cast<WhileStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::DoStmtClass:
    if (buildDoStmt(cast<DoStmt>(*S)).failed())
      return mlir::failure();
    break;

  case Stmt::CoroutineBodyStmtClass:
    return buildCoroutineBody(cast<CoroutineBodyStmt>(*S));
  case Stmt::CoreturnStmtClass:
    return buildCoreturnStmt(cast<CoreturnStmt>(*S));

  case Stmt::CXXForRangeStmtClass:
    return buildCXXForRangeStmt(cast<CXXForRangeStmt>(*S), Attrs);

  case Stmt::IndirectGotoStmtClass:
  case Stmt::ReturnStmtClass:
  // When implemented, GCCAsmStmtClass should fall-through to MSAsmStmtClass.
  case Stmt::GCCAsmStmtClass:
  case Stmt::MSAsmStmtClass:
  case Stmt::CapturedStmtClass:
  case Stmt::ObjCAtTryStmtClass:
  case Stmt::ObjCAtThrowStmtClass:
  case Stmt::ObjCAtSynchronizedStmtClass:
  case Stmt::ObjCForCollectionStmtClass:
  case Stmt::ObjCAutoreleasePoolStmtClass:
  case Stmt::CXXTryStmtClass:
  case Stmt::SEHTryStmtClass:
  case Stmt::OMPMetaDirectiveClass:
  case Stmt::OMPCanonicalLoopClass:
  case Stmt::OMPParallelDirectiveClass:
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
  case Stmt::OMPTaskyieldDirectiveClass:
  case Stmt::OMPBarrierDirectiveClass:
  case Stmt::OMPTaskwaitDirectiveClass:
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

mlir::LogicalResult CIRGenFunction::buildSimpleStmt(const Stmt *S,
                                                    bool useCurrentScope) {
  switch (S->getStmtClass()) {
  default:
    return mlir::failure();
  case Stmt::DeclStmtClass:
    return buildDeclStmt(cast<DeclStmt>(*S));
  case Stmt::CompoundStmtClass:
    return useCurrentScope
               ? buildCompoundStmtWithoutScope(cast<CompoundStmt>(*S))
               : buildCompoundStmt(cast<CompoundStmt>(*S));
  case Stmt::ReturnStmtClass:
    return buildReturnStmt(cast<ReturnStmt>(*S));
  case Stmt::GotoStmtClass:
    return buildGotoStmt(cast<GotoStmt>(*S));
  case Stmt::ContinueStmtClass:
    return buildContinueStmt(cast<ContinueStmt>(*S));

  case Stmt::NullStmtClass:
    break;

  case Stmt::LabelStmtClass:
    return buildLabelStmt(cast<LabelStmt>(*S));

  case Stmt::CaseStmtClass:
  case Stmt::DefaultStmtClass:
    assert(0 &&
           "Should not get here, currently handled directly from SwitchStmt");
    break;

  case Stmt::BreakStmtClass:
    return buildBreakStmt(cast<BreakStmt>(*S));

  case Stmt::AttributedStmtClass:
  case Stmt::SEHLeaveStmtClass:
    llvm::errs() << "CIR codegen for '" << S->getStmtClassName()
                 << "' not implemented\n";
    assert(0 && "not implemented");
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildLabelStmt(const clang::LabelStmt &S) {
  if (buildLabel(S.getDecl()).failed())
    return mlir::failure();

  // IsEHa: not implemented.
  assert(!(getContext().getLangOpts().EHAsynch && S.isSideEntry()));

  return buildStmt(S.getSubStmt(), /* useCurrentScope */ true);
}

// Add terminating yield on body regions (loops, ...) in case there are
// not other terminators used.
// FIXME: make terminateCaseRegion use this too.
static void terminateBody(mlir::OpBuilder &builder, mlir::Region &r,
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
      builder.create<YieldOp>(loc);
    }
  }

  for (auto *b : eraseBlocks)
    b->erase();
}

mlir::LogicalResult CIRGenFunction::buildIfStmt(const IfStmt &S) {
  // The else branch of a consteval if statement is always the only branch
  // that can be runtime evaluated.
  if (S.isConsteval()) {
    llvm_unreachable("consteval nyi");
  }
  mlir::LogicalResult res = mlir::success();

  // C99 6.8.4.1: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  auto ifStmtBuilder = [&]() -> mlir::LogicalResult {
    if (S.getInit())
      if (buildStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

    if (S.getConditionVariable())
      buildDecl(*S.getConditionVariable());

    // During LLVM codegen, if the condition constant folds and can be elided,
    // it tries to avoid emitting the condition and the dead arm of the if/else.
    // TODO(cir): we skip this in CIRGen, but should implement this as part of
    // SSCP or a specific CIR pass.
    bool CondConstant;
    if (ConstantFoldsToSimpleInteger(S.getCond(), CondConstant,
                                     S.isConstexpr())) {
      assert(!UnimplementedFeature::constantFoldsToSimpleInteger());
    }

    assert(!UnimplementedFeature::emitCondLikelihoodViaExpectIntrinsic());
    assert(!UnimplementedFeature::incrementProfileCounter());
    return buildIfOnBoolExpr(S.getCond(), S.getThen(), S.getElse());
  };

  // TODO: Add a new scoped symbol table.
  // LexicalScope ConditionScope(*this, S.getCond()->getSourceRange());
  // The if scope contains the full source range for IfStmt.
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScopeContext lexScope{scopeLoc, builder.getInsertionBlock()};
        LexicalScopeGuard lexIfScopeGuard{*this, &lexScope};
        res = ifStmtBuilder();
      });

  return res;
}

mlir::LogicalResult CIRGenFunction::buildDeclStmt(const DeclStmt &S) {
  if (!builder.getInsertionBlock()) {
    CGM.emitError("Seems like this is unreachable code, what should we do?");
    return mlir::failure();
  }

  for (const auto *I : S.decls()) {
    buildDecl(*I);
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildReturnStmt(const ReturnStmt &S) {
  assert(!UnimplementedFeature::requiresReturnValueCheck());
  auto loc = getLoc(S.getSourceRange());

  // Emit the result value, even if unused, to evaluate the side effects.
  const Expr *RV = S.getRetValue();

  // TODO(cir): LLVM codegen uses a RunCleanupsScope cleanupScope here, we
  // should model this in face of dtors.

  bool createNewScope = false;
  if (const auto *EWC = dyn_cast_or_null<ExprWithCleanups>(RV)) {
    RV = EWC->getSubExpr();
    createNewScope = true;
  }

  auto handleReturnVal = [&]() {
    if (getContext().getLangOpts().ElideConstructors && S.getNRVOCandidate() &&
        S.getNRVOCandidate()->isNRVOVariable()) {
      assert(!UnimplementedFeature::openMP());
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
        assert(0 && "not implemented");
      }
    } else if (!RV) {
      // Do nothing (return value is left uninitialized)
    } else if (FnRetTy->isReferenceType()) {
      // If this function returns a reference, take the address of the
      // expression rather than the value.
      RValue Result = buildReferenceBindingToExpr(RV);
      builder.create<mlir::cir::StoreOp>(loc, Result.getScalarVal(),
                                         ReturnValue.getPointer());
    } else {
      mlir::Value V = nullptr;
      switch (CIRGenFunction::getEvaluationKind(RV->getType())) {
      case TEK_Scalar:
        V = buildScalarExpr(RV);
        builder.create<mlir::cir::StoreOp>(loc, V, *FnRetAlloca);
        break;
      case TEK_Complex:
        llvm_unreachable("NYI");
        break;
      case TEK_Aggregate:
        buildAggExpr(
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
    builder.create<mlir::cir::ScopeOp>(
        scopeLoc, /*scopeBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          CIRGenFunction::LexicalScopeContext lexScope{
              loc, builder.getInsertionBlock()};
          CIRGenFunction::LexicalScopeGuard lexScopeGuard{*this, &lexScope};
          handleReturnVal();
        });
  }

  // Create a new return block (if not existent) and add a branch to
  // it. The actual return instruction is only inserted during current
  // scope cleanup handling.
  auto *retBlock = currLexScope->getOrCreateRetBlock(*this, loc);
  builder.create<BrOp>(loc, retBlock);

  // Insert the new block to continue codegen after branch to ret block.
  builder.createBlock(builder.getBlock()->getParent());

  // TODO(cir): LLVM codegen for a cleanup on cleanupScope here.
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildGotoStmt(const GotoStmt &S) {
  // FIXME: LLVM codegen inserts emit stop point here for debug info
  // sake when the insertion point is available, but doesn't do
  // anything special when there isn't. We haven't implemented debug
  // info support just yet, look at this again once we have it.
  assert(builder.getInsertionBlock() && "not yet implemented");

  // A goto marks the end of a block, create a new one for codegen after
  // buildGotoStmt can resume building in that block.

  // Build a cir.br to the target label.
  auto &JD = LabelMap[S.getLabel()];
  auto brOp = buildBranchThroughCleanup(getLoc(S.getSourceRange()), JD);
  if (!JD.isValid())
    currLexScope->PendingGotos.push_back(std::make_pair(brOp, S.getLabel()));

  // Insert the new block to continue codegen after goto.
  builder.createBlock(builder.getBlock()->getParent());

  // What here...
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildLabel(const LabelDecl *D) {
  JumpDest &Dest = LabelMap[D];

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
    builder.setInsertionPointToEnd(labelBlock);
  }

  if (!Dest.isValid()) {
    Dest.Block = labelBlock;
    currLexScope->SolvedLabels.insert(D);
    // FIXME: add a label attribute to block...
  } else {
    assert(0 && "unimplemented");
  }

  //  FIXME: emit debug info for labels, incrementProfileCounter
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::buildContinueStmt(const clang::ContinueStmt &S) {
  builder.create<YieldOp>(
      getLoc(S.getContinueLoc()),
      mlir::cir::YieldOpKindAttr::get(builder.getContext(),
                                      mlir::cir::YieldOpKind::Continue),
      mlir::ValueRange({}));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildBreakStmt(const clang::BreakStmt &S) {
  builder.create<YieldOp>(
      getLoc(S.getBreakLoc()),
      mlir::cir::YieldOpKindAttr::get(builder.getContext(),
                                      mlir::cir::YieldOpKind::Break),
      mlir::ValueRange({}));
  return mlir::success();
}

const CaseStmt *
CIRGenFunction::foldCaseStmt(const clang::CaseStmt &S, mlir::Type condType,
                             SmallVector<mlir::Attribute, 4> &caseAttrs) {
  const CaseStmt *caseStmt = &S;
  const CaseStmt *lastCase = &S;
  SmallVector<mlir::Attribute, 4> caseEltValueListAttr;

  // Fold cascading cases whenever possible to simplify codegen a bit.
  while (caseStmt) {
    lastCase = caseStmt;
    auto intVal = caseStmt->getLHS()->EvaluateKnownConstInt(getContext());
    caseEltValueListAttr.push_back(mlir::cir::IntAttr::get(condType, intVal));
    caseStmt = dyn_cast_or_null<CaseStmt>(caseStmt->getSubStmt());
  }

  auto *ctxt = builder.getContext();

  auto caseAttr = mlir::cir::CaseAttr::get(
      ctxt, builder.getArrayAttr(caseEltValueListAttr),
      CaseOpKindAttr::get(ctxt, caseEltValueListAttr.size() > 1
                                    ? mlir::cir::CaseOpKind::Anyof
                                    : mlir::cir::CaseOpKind::Equal));

  caseAttrs.push_back(caseAttr);

  return lastCase;
}

void CIRGenFunction::insertFallthrough(const clang::Stmt &S) {
  builder.create<YieldOp>(
      getLoc(S.getBeginLoc()),
      mlir::cir::YieldOpKindAttr::get(builder.getContext(),
                                      mlir::cir::YieldOpKind::Fallthrough),
      mlir::ValueRange({}));
}

template <typename T>
mlir::LogicalResult CIRGenFunction::buildCaseDefaultCascade(
    const T *stmt, mlir::Type condType,
    SmallVector<mlir::Attribute, 4> &caseAttrs, mlir::OperationState &os) {

  assert((isa<CaseStmt, DefaultStmt>(stmt)) &&
         "only case or default stmt go here");

  auto res = mlir::success();

  // Update scope information with the current region we are
  // emitting code for. This is useful to allow return blocks to be
  // automatically and properly placed during cleanup.
  auto *region = os.addRegion();
  auto *block = builder.createBlock(region);
  builder.setInsertionPointToEnd(block);
  currLexScope->updateCurrentSwitchCaseRegion();

  auto *sub = stmt->getSubStmt();

  if (isa<DefaultStmt>(sub) && isa<CaseStmt>(stmt)) {
    insertFallthrough(*stmt);
    res =
        buildDefaultStmt(*dyn_cast<DefaultStmt>(sub), condType, caseAttrs, os);
  } else if (isa<CaseStmt>(sub) && isa<DefaultStmt>(stmt)) {
    insertFallthrough(*stmt);
    res = buildCaseStmt(*dyn_cast<CaseStmt>(sub), condType, caseAttrs, os);
  } else {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    res = buildStmt(sub, /*useCurrentScope=*/!isa<CompoundStmt>(sub));
  }

  return res;
}

mlir::LogicalResult
CIRGenFunction::buildCaseStmt(const CaseStmt &S, mlir::Type condType,
                              SmallVector<mlir::Attribute, 4> &caseAttrs,
                              mlir::OperationState &os) {
  assert((!S.getRHS() || !S.caseStmtIsGNURange()) &&
         "case ranges not implemented");

  auto *caseStmt = foldCaseStmt(S, condType, caseAttrs);
  return buildCaseDefaultCascade(caseStmt, condType, caseAttrs, os);
}

mlir::LogicalResult
CIRGenFunction::buildDefaultStmt(const DefaultStmt &S, mlir::Type condType,
                                 SmallVector<mlir::Attribute, 4> &caseAttrs,
                                 mlir::OperationState &os) {
  auto ctxt = builder.getContext();

  auto defAttr = mlir::cir::CaseAttr::get(
      ctxt, builder.getArrayAttr({}),
      CaseOpKindAttr::get(ctxt, mlir::cir::CaseOpKind::Default));

  caseAttrs.push_back(defAttr);
  return buildCaseDefaultCascade(&S, condType, caseAttrs, os);
}

static mlir::LogicalResult buildLoopCondYield(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value cond) {
  mlir::Block *trueBB = nullptr, *falseBB = nullptr;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    trueBB = builder.createBlock(builder.getBlock()->getParent());
    builder.create<mlir::cir::YieldOp>(loc, YieldOpKind::Continue);
  }
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    falseBB = builder.createBlock(builder.getBlock()->getParent());
    builder.create<mlir::cir::YieldOp>(loc);
  }

  assert((trueBB && falseBB) && "expected both blocks to exist");
  builder.create<mlir::cir::BrCondOp>(loc, cond, trueBB, falseBB);
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::buildCXXForRangeStmt(const CXXForRangeStmt &S,
                                     ArrayRef<const Attr *> ForAttrs) {
  mlir::cir::LoopOp loopOp;

  // TODO(cir): pass in array of attributes.
  auto forStmtBuilder = [&]() -> mlir::LogicalResult {
    auto loopRes = mlir::success();
    // Evaluate the first pieces before the loop.
    if (S.getInit())
      if (buildStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();
    if (buildStmt(S.getRangeStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();
    if (buildStmt(S.getBeginStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();
    if (buildStmt(S.getEndStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    assert(!UnimplementedFeature::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!UnimplementedFeature::requiresCleanups());

    loopOp = builder.create<LoopOp>(
        getLoc(S.getSourceRange()), mlir::cir::LoopOpKind::For,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!UnimplementedFeature::createProfileWeightsForLoop());
          assert(!UnimplementedFeature::emitCondLikelihoodViaExpectIntrinsic());
          mlir::Value condVal = evaluateExprAsBool(S.getCond());
          if (buildLoopCondYield(b, loc, condVal).failed())
            loopRes = mlir::failure();
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // https://en.cppreference.com/w/cpp/language/for
          // In C++ the scope of the init-statement and the scope of
          // statement are one and the same.
          bool useCurrentScope = true;
          if (buildStmt(S.getLoopVarStmt(), useCurrentScope).failed())
            loopRes = mlir::failure();
          if (buildStmt(S.getBody(), useCurrentScope).failed())
            loopRes = mlir::failure();
          buildStopPoint(&S);
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (S.getInc())
            if (buildStmt(S.getInc(), /*useCurrentScope=*/true).failed())
              loopRes = mlir::failure();
          builder.create<YieldOp>(loc);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        // Create a cleanup scope for the condition variable cleanups.
        // Logical equivalent from LLVM codegn for
        // LexicalScope ConditionScope(*this, S.getSourceRange())...
        LexicalScopeContext lexScope{loc, builder.getInsertionBlock()};
        LexicalScopeGuard lexForScopeGuard{*this, &lexScope};
        res = forStmtBuilder();
      });

  if (res.failed())
    return res;

  terminateBody(builder, loopOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildForStmt(const ForStmt &S) {
  mlir::cir::LoopOp loopOp;

  // TODO: pass in array of attributes.
  auto forStmtBuilder = [&]() -> mlir::LogicalResult {
    auto loopRes = mlir::success();
    // Evaluate the first part before the loop.
    if (S.getInit())
      if (buildStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();
    assert(!UnimplementedFeature::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!UnimplementedFeature::requiresCleanups());

    loopOp = builder.create<LoopOp>(
        getLoc(S.getSourceRange()), mlir::cir::LoopOpKind::For,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!UnimplementedFeature::createProfileWeightsForLoop());
          assert(!UnimplementedFeature::emitCondLikelihoodViaExpectIntrinsic());
          mlir::Value condVal;
          if (S.getCond()) {
            // If the for statement has a condition scope,
            // emit the local variable declaration.
            if (S.getConditionVariable())
              buildDecl(*S.getConditionVariable());
            // C99 6.8.5p2/p4: The first substatement is executed if the
            // expression compares unequal to 0. The condition must be a
            // scalar type.
            condVal = evaluateExprAsBool(S.getCond());
          } else {
            auto boolTy = mlir::cir::BoolType::get(b.getContext());
            condVal = b.create<mlir::cir::ConstantOp>(
                loc, boolTy,
                mlir::cir::BoolAttr::get(b.getContext(), boolTy, true));
          }
          if (buildLoopCondYield(b, loc, condVal).failed())
            loopRes = mlir::failure();
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // https://en.cppreference.com/w/cpp/language/for
          // While in C++, the scope of the init-statement and the scope of
          // statement are one and the same, in C the scope of statement is
          // nested within the scope of init-statement.
          bool useCurrentScope =
              CGM.getASTContext().getLangOpts().CPlusPlus ? true : false;
          if (buildStmt(S.getBody(), useCurrentScope).failed())
            loopRes = mlir::failure();
          buildStopPoint(&S);
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (S.getInc())
            if (buildStmt(S.getInc(), /*useCurrentScope=*/true).failed())
              loopRes = mlir::failure();
          builder.create<YieldOp>(loc);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScopeContext lexScope{loc, builder.getInsertionBlock()};
        LexicalScopeGuard lexForScopeGuard{*this, &lexScope};
        res = forStmtBuilder();
      });

  if (res.failed())
    return res;

  terminateBody(builder, loopOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildDoStmt(const DoStmt &S) {
  mlir::cir::LoopOp loopOp;

  // TODO: pass in array of attributes.
  auto doStmtBuilder = [&]() -> mlir::LogicalResult {
    auto loopRes = mlir::success();
    assert(!UnimplementedFeature::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!UnimplementedFeature::requiresCleanups());

    loopOp = builder.create<LoopOp>(
        getLoc(S.getSourceRange()), mlir::cir::LoopOpKind::DoWhile,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!UnimplementedFeature::createProfileWeightsForLoop());
          assert(!UnimplementedFeature::emitCondLikelihoodViaExpectIntrinsic());
          // C99 6.8.5p2/p4: The first substatement is executed if the
          // expression compares unequal to 0. The condition must be a
          // scalar type.
          mlir::Value condVal = evaluateExprAsBool(S.getCond());
          if (buildLoopCondYield(b, loc, condVal).failed())
            loopRes = mlir::failure();
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (buildStmt(S.getBody(), /*useCurrentScope=*/true).failed())
            loopRes = mlir::failure();
          buildStopPoint(&S);
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          builder.create<YieldOp>(loc);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScopeContext lexScope{loc, builder.getInsertionBlock()};
        LexicalScopeGuard lexForScopeGuard{*this, &lexScope};
        res = doStmtBuilder();
      });

  if (res.failed())
    return res;

  terminateBody(builder, loopOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildWhileStmt(const WhileStmt &S) {
  mlir::cir::LoopOp loopOp;

  // TODO: pass in array of attributes.
  auto whileStmtBuilder = [&]() -> mlir::LogicalResult {
    auto loopRes = mlir::success();
    assert(!UnimplementedFeature::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!UnimplementedFeature::requiresCleanups());

    loopOp = builder.create<LoopOp>(
        getLoc(S.getSourceRange()), mlir::cir::LoopOpKind::While,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!UnimplementedFeature::createProfileWeightsForLoop());
          assert(!UnimplementedFeature::emitCondLikelihoodViaExpectIntrinsic());
          mlir::Value condVal;
          // If the for statement has a condition scope,
          // emit the local variable declaration.
          if (S.getConditionVariable())
            buildDecl(*S.getConditionVariable());
          // C99 6.8.5p2/p4: The first substatement is executed if the
          // expression compares unequal to 0. The condition must be a
          // scalar type.
          condVal = evaluateExprAsBool(S.getCond());
          if (buildLoopCondYield(b, loc, condVal).failed())
            loopRes = mlir::failure();
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (buildStmt(S.getBody(), /*useCurrentScope=*/true).failed())
            loopRes = mlir::failure();
          buildStopPoint(&S);
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          builder.create<YieldOp>(loc);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScopeContext lexScope{loc, builder.getInsertionBlock()};
        LexicalScopeGuard lexForScopeGuard{*this, &lexScope};
        res = whileStmtBuilder();
      });

  if (res.failed())
    return res;

  terminateBody(builder, loopOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildSwitchStmt(const SwitchStmt &S) {
  // TODO: LLVM codegen does some early optimization to fold the condition and
  // only emit live cases. CIR should use MLIR to achieve similar things,
  // nothing to be done here.
  // if (ConstantFoldsToSimpleInteger(S.getCond(), ConstantCondValue))...

  auto res = mlir::success();
  SwitchOp swop;

  auto switchStmtBuilder = [&]() -> mlir::LogicalResult {
    if (S.getInit())
      if (buildStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

    if (S.getConditionVariable())
      buildDecl(*S.getConditionVariable());

    mlir::Value condV = buildScalarExpr(S.getCond());

    // TODO: PGO and likelihood (e.g. PGO.haveRegionCounts())
    // TODO: if the switch has a condition wrapped by __builtin_unpredictable?

    // FIXME: track switch to handle nested stmts.
    swop = builder.create<SwitchOp>(
        getLoc(S.getBeginLoc()), condV,
        /*switchBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::OperationState &os) {
          auto *cs = dyn_cast<CompoundStmt>(S.getBody());
          assert(cs && "expected compound stmt");
          SmallVector<mlir::Attribute, 4> caseAttrs;

          currLexScope->setAsSwitch();
          mlir::Block *lastCaseBlock = nullptr;
          for (auto *c : cs->body()) {
            bool caseLike = isa<CaseStmt, DefaultStmt>(c);
            if (!caseLike) {
              // This means it's a random stmt following up a case, just
              // emit it as part of previous known case.
              assert(lastCaseBlock && "expects pre-existing case block");
              mlir::OpBuilder::InsertionGuard guardCase(builder);
              builder.setInsertionPointToEnd(lastCaseBlock);
              res = buildStmt(c, /*useCurrentScope=*/!isa<CompoundStmt>(c));
              if (res.failed())
                break;
              continue;
            }

            auto *caseStmt = dyn_cast<CaseStmt>(c);

            if (caseStmt)
              res = buildCaseStmt(*caseStmt, condV.getType(), caseAttrs, os);
            else {
              auto *defaultStmt = dyn_cast<DefaultStmt>(c);
              assert(defaultStmt && "expected default stmt");
              res = buildDefaultStmt(*defaultStmt, condV.getType(), caseAttrs,
                                     os);
            }

            lastCaseBlock = builder.getBlock();

            if (res.failed())
              break;
          }

          os.addAttribute("cases", builder.getArrayAttr(caseAttrs));
        });

    if (res.failed())
      return res;
    return mlir::success();
  };

  // The switch scope contains the full source range for SwitchStmt.
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScopeContext lexScope{loc, builder.getInsertionBlock()};
        LexicalScopeGuard lexIfScopeGuard{*this, &lexScope};
        res = switchStmtBuilder();
      });

  if (res.failed())
    return res;

  // Any block in a case region without a terminator is considered a
  // fallthrough yield. In practice there shouldn't be more than one
  // block without a terminator, we patch any block we see though and
  // let mlir's SwitchOp verifier enforce rules.
  auto terminateCaseRegion = [&](mlir::Region &r, mlir::Location loc) {
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
        builder.create<YieldOp>(
            loc,
            mlir::cir::YieldOpKindAttr::get(
                builder.getContext(), mlir::cir::YieldOpKind::Fallthrough),
            mlir::ValueRange({}));
      }
    }

    for (auto *b : eraseBlocks)
      b->erase();
  };

  // Make sure all case regions are terminated by inserting fallthroughs
  // when necessary.
  // FIXME: find a better way to get accurante with location here.
  for (auto &r : swop.getRegions())
    terminateCaseRegion(r, swop.getLoc());
  return mlir::success();
}

void CIRGenFunction::buildReturnOfRValue(mlir::Location loc, RValue RV,
                                         QualType Ty) {
  if (RV.isScalar()) {
    builder.createStore(loc, RV.getScalarVal(), ReturnValue);
  } else if (RV.isAggregate()) {
    LValue Dest = makeAddrLValue(ReturnValue, Ty);
    LValue Src = makeAddrLValue(RV.getAggregateAddress(), Ty);
    buildAggregateCopy(Dest, Src, Ty, getOverlapForReturnValue());
  } else {
    llvm_unreachable("NYI");
  }
  buildBranchThroughCleanup(loc, ReturnBlock());
}
