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
#include "clang/AST/StmtOpenACC.h"

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
  mlir::OpBuilder::InsertPoint scopeInsPt;
  builder.create<cir::ScopeOp>(
      scopeLoc, [&](mlir::OpBuilder &b, mlir::Type &type, mlir::Location loc) {
        scopeInsPt = b.saveInsertionPoint();
      });
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.restoreInsertionPoint(scopeInsPt);
    LexicalScope lexScope(*this, scopeLoc, builder.getInsertionBlock());
    emitCompoundStmtWithoutScope(s);
  }
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
  case Stmt::BreakStmtClass:
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
      // Remember the block we came in on.
      mlir::Block *incoming = builder.getInsertionBlock();
      assert(incoming && "expression emission must have an insertion point");

      emitIgnoredExpr(cast<Expr>(s));

      mlir::Block *outgoing = builder.getInsertionBlock();
      assert(outgoing && "expression emission cleared block!");
      return mlir::success();
    }

  case Stmt::ForStmtClass:
    return emitForStmt(cast<ForStmt>(*s));
  case Stmt::WhileStmtClass:
    return emitWhileStmt(cast<WhileStmt>(*s));
  case Stmt::DoStmtClass:
    return emitDoStmt(cast<DoStmt>(*s));
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
  case Stmt::OMPScopeDirectiveClass:
  case Stmt::OMPErrorDirectiveClass:
  case Stmt::NoStmtClass:
  case Stmt::CXXCatchStmtClass:
  case Stmt::SEHExceptStmtClass:
  case Stmt::SEHFinallyStmtClass:
  case Stmt::MSDependentExistsStmtClass:
  case Stmt::NullStmtClass:
  case Stmt::LabelStmtClass:
  case Stmt::AttributedStmtClass:
  case Stmt::GotoStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::CaseStmtClass:
  case Stmt::SEHLeaveStmtClass:
  case Stmt::SYCLKernelCallStmtClass:
  case Stmt::IfStmtClass:
  case Stmt::SwitchStmtClass:
  case Stmt::CoroutineBodyStmtClass:
  case Stmt::CoreturnStmtClass:
  case Stmt::CXXTryStmtClass:
  case Stmt::CXXForRangeStmtClass:
  case Stmt::IndirectGotoStmtClass:
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
      emitCompoundStmtWithoutScope(cast<CompoundStmt>(*s));
    else
      emitCompoundStmt(cast<CompoundStmt>(*s));
    break;
  case Stmt::ContinueStmtClass:
    return emitContinueStmt(cast<ContinueStmt>(*s));
  case Stmt::BreakStmtClass:
    return emitBreakStmt(cast<BreakStmt>(*s));
  case Stmt::ReturnStmtClass:
    return emitReturnStmt(cast<ReturnStmt>(*s));
  }

  return mlir::success();
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
  } else if (rv->getType()->isVoidType()) {
    // Make sure not to return anything, but evaluate the expression
    // for side effects.
    if (rv) {
      emitAnyExpr(rv);
    }
  } else if (cast<FunctionDecl>(curGD.getDecl())
                 ->getReturnType()
                 ->isReferenceType()) {
    getCIRGenModule().errorNYI(s.getSourceRange(),
                               "function return type that is a reference");
  } else {
    mlir::Value value = nullptr;
    switch (CIRGenFunction::getEvaluationKind(rv->getType())) {
    case cir::TEK_Scalar:
      value = emitScalarExpr(rv);
      if (value) { // Change this to an assert once emitScalarExpr is complete
        builder.CIRBaseBuilderTy::createStore(loc, value, *fnRetAlloca);
      }
      break;
    default:
      getCIRGenModule().errorNYI(s.getSourceRange(),
                                 "non-scalar function return type");
      break;
    }
  }

  auto *retBlock = curLexScope->getOrCreateRetBlock(*this, loc);
  builder.create<cir::BrOp>(loc, retBlock);
  builder.createBlock(builder.getBlock()->getParent());

  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitContinueStmt(const clang::ContinueStmt &s) {
  builder.createContinue(getLoc(s.getContinueLoc()));

  // Insert the new block to continue codegen after the continue statement.
  builder.createBlock(builder.getBlock()->getParent());

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitBreakStmt(const clang::BreakStmt &s) {
  builder.createBreak(getLoc(s.getBreakLoc()));

  // Insert the new block to continue codegen after the break statement.
  builder.createBlock(builder.getBlock()->getParent());

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
            cir::BoolType boolTy = cir::BoolType::get(b.getContext());
            condVal = b.create<cir::ConstantOp>(
                loc, boolTy, cir::BoolAttr::get(b.getContext(), boolTy, true));
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
