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

using namespace cir;
using namespace clang;
using namespace mlir::cir;

Address CIRGenFunction::buildCompoundStmtWithoutScope(const CompoundStmt &S,
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
        buildAggExpr(E, slot);
      } else {
        // We can't return an RValue here because there might be cleanups at
        // the end of the StmtExpr.  Because of that, we have to emit the result
        // here into a temporary alloca.
        retAlloca = CreateMemTemp(exprTy, getLoc(E->getSourceRange()));
        buildAnyExprToMem(E, retAlloca, Qualifiers(),
                          /*IsInit*/ false);
      }
    } else {
      if (buildStmt(CurStmt, /*useCurrentScope=*/false).failed())
        llvm_unreachable("failed to build statement");
    }
  }

  return retAlloca;
}

Address CIRGenFunction::buildCompoundStmt(const CompoundStmt &S, bool getLast,
                                          AggValueSlot slot) {
  Address retAlloca = Address::invalid();

  // Add local scope to track new declared variables.
  SymTableScopeTy varScope(symbolTable);
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Type &type, mlir::Location loc) {
        LexicalScope lexScope{*this, loc, builder.getInsertionBlock()};
        retAlloca = buildCompoundStmtWithoutScope(S, getLast, slot);
      });

  return retAlloca;
}

void CIRGenFunction::buildStopPoint(const Stmt *S) {
  assert(!MissingFeatures::generateDebugInfo());
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

      buildIgnoredExpr(cast<Expr>(S));

      mlir::Block *outgoing = builder.getInsertionBlock();
      assert(outgoing && "expression emission cleared block!");

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

  case Stmt::CXXTryStmtClass:
    return buildCXXTryStmt(cast<CXXTryStmt>(*S));

  case Stmt::CXXForRangeStmtClass:
    return buildCXXForRangeStmt(cast<CXXForRangeStmt>(*S), Attrs);

  case Stmt::IndirectGotoStmtClass:
  case Stmt::ReturnStmtClass:
  // When implemented, GCCAsmStmtClass should fall-through to MSAsmStmtClass.
  case Stmt::GCCAsmStmtClass:
  case Stmt::MSAsmStmtClass:
    return buildAsmStmt(cast<AsmStmt>(*S));
  // OMP directives:
  case Stmt::OMPParallelDirectiveClass:
    return buildOMPParallelDirective(cast<OMPParallelDirective>(*S));
  case Stmt::OMPTaskwaitDirectiveClass:
    return buildOMPTaskwaitDirective(cast<OMPTaskwaitDirective>(*S));
  case Stmt::OMPTaskyieldDirectiveClass:
    return buildOMPTaskyieldDirective(cast<OMPTaskyieldDirective>(*S));
  case Stmt::OMPBarrierDirectiveClass:
    return buildOMPBarrierDirective(cast<OMPBarrierDirective>(*S));
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
    useCurrentScope ? buildCompoundStmtWithoutScope(cast<CompoundStmt>(*S))
                    : buildCompoundStmt(cast<CompoundStmt>(*S));
    break;
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
    return buildSwitchCase(cast<SwitchCase>(*S));
    break;

  case Stmt::BreakStmtClass:
    return buildBreakStmt(cast<BreakStmt>(*S));

  case Stmt::AttributedStmtClass:
    return buildAttributedStmt(cast<AttributedStmt>(*S));

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

mlir::LogicalResult
CIRGenFunction::buildAttributedStmt(const AttributedStmt &S) {
  for (const auto *A : S.getAttrs()) {
    switch (A->getKind()) {
    case attr::NoMerge:
    case attr::NoInline:
    case attr::AlwaysInline:
    case attr::MustTail:
      llvm_unreachable("NIY attributes");
    default:
      break;
    }
  }

  return buildStmt(S.getSubStmt(), true, S.getAttrs());
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

mlir::LogicalResult CIRGenFunction::buildIfStmt(const IfStmt &S) {
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
      return buildStmt(ConstevalExecuted, /*useCurrentScope=*/true);

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
      if (S.isConstexpr()) {
        // Handle "if constexpr" explicitly here to avoid generating some
        // ill-formed code since in CIR the "if" is no longer simplified
        // in this lambda like in Clang but postponed to other MLIR
        // passes.
        if (const Stmt *Executed = CondConstant ? S.getThen() : S.getElse())
          return buildStmt(Executed, /*useCurrentScope=*/true);
        // There is nothing to execute at runtime.
        // TODO(cir): there is still an empty cir.scope generated by the caller.
        return mlir::success();
      }
      assert(!MissingFeatures::constantFoldsToSimpleInteger());
    }

    assert(!MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
    assert(!MissingFeatures::incrementProfileCounter());
    return buildIfOnBoolExpr(S.getCond(), S.getThen(), S.getElse());
  };

  // TODO: Add a new scoped symbol table.
  // LexicalScope ConditionScope(*this, S.getCond()->getSourceRange());
  // The if scope contains the full source range for IfStmt.
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, scopeLoc, builder.getInsertionBlock()};
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
  assert(!MissingFeatures::requiresReturnValueCheck());
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
      assert(!MissingFeatures::openMP());
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
        buildAnyExpr(RV);
      }
    } else if (!RV) {
      // Do nothing (return value is left uninitialized)
    } else if (FnRetTy->isReferenceType()) {
      // If this function returns a reference, take the address of the
      // expression rather than the value.
      RValue Result = buildReferenceBindingToExpr(RV);
      builder.createStore(loc, Result.getScalarVal(), ReturnValue);
    } else {
      mlir::Value V = nullptr;
      switch (CIRGenFunction::getEvaluationKind(RV->getType())) {
      case TEK_Scalar:
        V = buildScalarExpr(RV);
        builder.CIRBaseBuilderTy::createStore(loc, V, *FnRetAlloca);
        break;
      case TEK_Complex:
        buildComplexExprIntoLValue(RV,
                                   makeAddrLValue(ReturnValue, RV->getType()),
                                   /*isInit*/ true);
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
          CIRGenFunction::LexicalScope lexScope{*this, loc,
                                                builder.getInsertionBlock()};
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

  builder.create<mlir::cir::GotoOp>(getLoc(S.getSourceRange()),
                                    S.getLabel()->getName());

  // A goto marks the end of a block, create a new one for codegen after
  // buildGotoStmt can resume building in that block.
  // Insert the new block to continue codegen after goto.
  builder.createBlock(builder.getBlock()->getParent());

  // What here...
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildLabel(const LabelDecl *D) {
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
  builder.create<mlir::cir::LabelOp>(getLoc(D->getSourceRange()), D->getName());
  builder.setInsertionPointToEnd(labelBlock);

  //  FIXME: emit debug info for labels, incrementProfileCounter
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::buildContinueStmt(const clang::ContinueStmt &S) {
  builder.createContinue(getLoc(S.getContinueLoc()));

  // Insert the new block to continue codegen after the continue statement.
  builder.createBlock(builder.getBlock()->getParent());

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildBreakStmt(const clang::BreakStmt &S) {
  builder.createBreak(getLoc(S.getBreakLoc()));

  // Insert the new block to continue codegen after the break statement.
  builder.createBlock(builder.getBlock()->getParent());

  return mlir::success();
}

const CaseStmt *
CIRGenFunction::foldCaseStmt(const clang::CaseStmt &S, mlir::Type condType,
                             SmallVector<mlir::Attribute, 4> &caseAttrs) {
  auto *ctxt = builder.getContext();

  const CaseStmt *caseStmt = &S;
  const CaseStmt *lastCase = &S;
  SmallVector<mlir::Attribute, 4> caseEltValueListAttr;

  int caseAttrCount = 0;

  // Fold cascading cases whenever possible to simplify codegen a bit.
  while (caseStmt) {
    lastCase = caseStmt;

    auto intVal = caseStmt->getLHS()->EvaluateKnownConstInt(getContext());

    if (auto *rhs = caseStmt->getRHS()) {
      auto endVal = rhs->EvaluateKnownConstInt(getContext());
      SmallVector<mlir::Attribute, 4> rangeCaseAttr = {
          mlir::cir::IntAttr::get(condType, intVal),
          mlir::cir::IntAttr::get(condType, endVal)};
      auto caseAttr = mlir::cir::CaseAttr::get(
          ctxt, builder.getArrayAttr(rangeCaseAttr),
          CaseOpKindAttr::get(ctxt, mlir::cir::CaseOpKind::Range));
      caseAttrs.push_back(caseAttr);
      ++caseAttrCount;
    } else {
      caseEltValueListAttr.push_back(mlir::cir::IntAttr::get(condType, intVal));
    }

    caseStmt = dyn_cast_or_null<CaseStmt>(caseStmt->getSubStmt());
  }

  if (!caseEltValueListAttr.empty()) {
    auto caseOpKind = caseEltValueListAttr.size() > 1
                          ? mlir::cir::CaseOpKind::Anyof
                          : mlir::cir::CaseOpKind::Equal;
    auto caseAttr = mlir::cir::CaseAttr::get(
        ctxt, builder.getArrayAttr(caseEltValueListAttr),
        CaseOpKindAttr::get(ctxt, caseOpKind));
    caseAttrs.push_back(caseAttr);
    ++caseAttrCount;
  }

  assert(caseAttrCount > 0 && "there should be at least one valid case attr");

  for (int i = 1; i < caseAttrCount; ++i) {
    // If there are multiple case attributes, we need to create a new region
    auto *region = currLexScope->createSwitchRegion();
    builder.createBlock(region);
  }

  return lastCase;
}

template <typename T>
mlir::LogicalResult CIRGenFunction::buildCaseDefaultCascade(
    const T *stmt, mlir::Type condType,
    SmallVector<mlir::Attribute, 4> &caseAttrs) {

  assert((isa<CaseStmt, DefaultStmt>(stmt)) &&
         "only case or default stmt go here");

  auto res = mlir::success();

  // Update scope information with the current region we are
  // emitting code for. This is useful to allow return blocks to be
  // automatically and properly placed during cleanup.
  auto *region = currLexScope->createSwitchRegion();
  auto *block = builder.createBlock(region);
  builder.setInsertionPointToEnd(block);

  auto *sub = stmt->getSubStmt();

  if (isa<DefaultStmt>(sub) && isa<CaseStmt>(stmt)) {
    builder.createYield(getLoc(stmt->getBeginLoc()));
    res = buildDefaultStmt(*dyn_cast<DefaultStmt>(sub), condType, caseAttrs);
  } else if (isa<CaseStmt>(sub) && isa<DefaultStmt>(stmt)) {
    builder.createYield(getLoc(stmt->getBeginLoc()));
    res = buildCaseStmt(*dyn_cast<CaseStmt>(sub), condType, caseAttrs);
  } else {
    res = buildStmt(sub, /*useCurrentScope=*/!isa<CompoundStmt>(sub));
  }

  return res;
}

mlir::LogicalResult
CIRGenFunction::buildCaseStmt(const CaseStmt &S, mlir::Type condType,
                              SmallVector<mlir::Attribute, 4> &caseAttrs) {
  auto *caseStmt = foldCaseStmt(S, condType, caseAttrs);
  return buildCaseDefaultCascade(caseStmt, condType, caseAttrs);
}

mlir::LogicalResult
CIRGenFunction::buildDefaultStmt(const DefaultStmt &S, mlir::Type condType,
                                 SmallVector<mlir::Attribute, 4> &caseAttrs) {
  auto ctxt = builder.getContext();

  auto defAttr = mlir::cir::CaseAttr::get(
      ctxt, builder.getArrayAttr({}),
      CaseOpKindAttr::get(ctxt, mlir::cir::CaseOpKind::Default));

  caseAttrs.push_back(defAttr);
  return buildCaseDefaultCascade(&S, condType, caseAttrs);
}

mlir::LogicalResult CIRGenFunction::buildSwitchCase(const SwitchCase &S) {
  assert(!caseAttrsStack.empty() &&
         "build switch case without seeting case attrs");
  assert(!condTypeStack.empty() &&
         "build switch case without specifying the type of the condition");

  if (S.getStmtClass() == Stmt::CaseStmtClass)
    return buildCaseStmt(cast<CaseStmt>(S), condTypeStack.back(),
                         caseAttrsStack.back());

  if (S.getStmtClass() == Stmt::DefaultStmtClass)
    return buildDefaultStmt(cast<DefaultStmt>(S), condTypeStack.back(),
                            caseAttrsStack.back());

  llvm_unreachable("expect case or default stmt");
}

mlir::LogicalResult
CIRGenFunction::buildCXXForRangeStmt(const CXXForRangeStmt &S,
                                     ArrayRef<const Attr *> ForAttrs) {
  mlir::cir::ForOp forOp;

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

    assert(!MissingFeatures::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!MissingFeatures::requiresCleanups());

    forOp = builder.createFor(
        getLoc(S.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!MissingFeatures::createProfileWeightsForLoop());
          assert(!MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          mlir::Value condVal = evaluateExprAsBool(S.getCond());
          builder.createCondition(condVal);
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
          builder.createYield(loc);
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
        LexicalScope lexScope{*this, loc, builder.getInsertionBlock()};
        res = forStmtBuilder();
      });

  if (res.failed())
    return res;

  terminateBody(builder, forOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildForStmt(const ForStmt &S) {
  mlir::cir::ForOp forOp;

  // TODO: pass in array of attributes.
  auto forStmtBuilder = [&]() -> mlir::LogicalResult {
    auto loopRes = mlir::success();
    // Evaluate the first part before the loop.
    if (S.getInit())
      if (buildStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();
    assert(!MissingFeatures::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!MissingFeatures::requiresCleanups());

    forOp = builder.createFor(
        getLoc(S.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!MissingFeatures::createProfileWeightsForLoop());
          assert(!MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
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
          builder.createCondition(condVal);
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
          builder.createYield(loc);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, loc, builder.getInsertionBlock()};
        res = forStmtBuilder();
      });

  if (res.failed())
    return res;

  terminateBody(builder, forOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildDoStmt(const DoStmt &S) {
  mlir::cir::DoWhileOp doWhileOp;

  // TODO: pass in array of attributes.
  auto doStmtBuilder = [&]() -> mlir::LogicalResult {
    auto loopRes = mlir::success();
    assert(!MissingFeatures::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!MissingFeatures::requiresCleanups());

    doWhileOp = builder.createDoWhile(
        getLoc(S.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!MissingFeatures::createProfileWeightsForLoop());
          assert(!MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          // C99 6.8.5p2/p4: The first substatement is executed if the
          // expression compares unequal to 0. The condition must be a
          // scalar type.
          mlir::Value condVal = evaluateExprAsBool(S.getCond());
          builder.createCondition(condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (buildStmt(S.getBody(), /*useCurrentScope=*/true).failed())
            loopRes = mlir::failure();
          buildStopPoint(&S);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, loc, builder.getInsertionBlock()};
        res = doStmtBuilder();
      });

  if (res.failed())
    return res;

  terminateBody(builder, doWhileOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildWhileStmt(const WhileStmt &S) {
  mlir::cir::WhileOp whileOp;

  // TODO: pass in array of attributes.
  auto whileStmtBuilder = [&]() -> mlir::LogicalResult {
    auto loopRes = mlir::success();
    assert(!MissingFeatures::loopInfoStack());
    // From LLVM: if there are any cleanups between here and the loop-exit
    // scope, create a block to stage a loop exit along.
    // We probably already do the right thing because of ScopeOp, but make
    // sure we handle all cases.
    assert(!MissingFeatures::requiresCleanups());

    whileOp = builder.createWhile(
        getLoc(S.getSourceRange()),
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          assert(!MissingFeatures::createProfileWeightsForLoop());
          assert(!MissingFeatures::emitCondLikelihoodViaExpectIntrinsic());
          mlir::Value condVal;
          // If the for statement has a condition scope,
          // emit the local variable declaration.
          if (S.getConditionVariable())
            buildDecl(*S.getConditionVariable());
          // C99 6.8.5p2/p4: The first substatement is executed if the
          // expression compares unequal to 0. The condition must be a
          // scalar type.
          condVal = evaluateExprAsBool(S.getCond());
          builder.createCondition(condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (buildStmt(S.getBody(), /*useCurrentScope=*/true).failed())
            loopRes = mlir::failure();
          buildStopPoint(&S);
        });
    return loopRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        LexicalScope lexScope{*this, loc, builder.getInsertionBlock()};
        res = whileStmtBuilder();
      });

  if (res.failed())
    return res;

  terminateBody(builder, whileOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildSwitchBody(const Stmt *S) {
  if (auto *compoundStmt = dyn_cast<CompoundStmt>(S)) {
    mlir::Block *lastCaseBlock = nullptr;
    auto res = mlir::success();
    for (auto *c : compoundStmt->body()) {
      if (auto *switchCase = dyn_cast<SwitchCase>(c)) {
        res = buildSwitchCase(*switchCase);
        lastCaseBlock = builder.getBlock();
      } else if (lastCaseBlock) {
        // This means it's a random stmt following up a case, just
        // emit it as part of previous known case.
        mlir::OpBuilder::InsertionGuard guardCase(builder);
        builder.setInsertionPointToEnd(lastCaseBlock);
        res = buildStmt(c, /*useCurrentScope=*/!isa<CompoundStmt>(c));
        lastCaseBlock = builder.getBlock();
      } else {
        llvm_unreachable("statement doesn't belong to any case region, NYI");
      }

      if (res.failed())
        break;
    }
    return res;
  }

  llvm_unreachable("switch body is not CompoundStmt, NYI");
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

    swop = builder.create<SwitchOp>(
        getLoc(S.getBeginLoc()), condV,
        /*switchBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::OperationState &os) {
          currLexScope->setAsSwitch();

          caseAttrsStack.push_back({});
          condTypeStack.push_back(condV.getType());

          res = buildSwitchBody(S.getBody());

          os.addRegions(currLexScope->getSwitchRegions());
          os.addAttribute("cases", builder.getArrayAttr(caseAttrsStack.back()));

          caseAttrsStack.pop_back();
          condTypeStack.pop_back();
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
        LexicalScope lexScope{*this, loc, builder.getInsertionBlock()};
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
        builder.createYield(loc);
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
