//===- ExprEngineCXX.cpp - ExprEngine support for C++ -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the C++ expression evaluation engine.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/PrettyStackTrace.h"

using namespace clang;
using namespace ento;

void ExprEngine::CreateCXXTemporaryObject(const MaterializeTemporaryExpr *ME,
                                          ExplodedNode *Pred,
                                          ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  const Expr *tempExpr = ME->GetTemporaryExpr()->IgnoreParens();
  ProgramStateRef state = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();

  // Bind the temporary object to the value of the expression. Then bind
  // the expression to the location of the object.
  SVal V = state->getSVal(tempExpr, LCtx);

  // If the value is already a CXXTempObjectRegion, it is fine as it is.
  // Otherwise, create a new CXXTempObjectRegion, and copy the value into it.
  const MemRegion *MR = V.getAsRegion();
  if (!MR || !isa<CXXTempObjectRegion>(MR)) {
    const MemRegion *R =
      svalBuilder.getRegionManager().getCXXTempObjectRegion(ME, LCtx);

    SVal L = loc::MemRegionVal(R);
    state = state->bindLoc(L, V);
    V = L;
  }

  Bldr.generateNode(ME, Pred, state->BindExpr(ME, LCtx, V));
}

void ExprEngine::VisitCXXConstructExpr(const CXXConstructExpr *CE,
                                       ExplodedNode *Pred,
                                       ExplodedNodeSet &destNodes) {
  const LocationContext *LCtx = Pred->getLocationContext();
  ProgramStateRef State = Pred->getState();

  const MemRegion *Target = 0;

  switch (CE->getConstructionKind()) {
  case CXXConstructExpr::CK_Complete: {
    // See if we're constructing an existing region by looking at the next
    // element in the CFG.
    const CFGBlock *B = currBldrCtx->getBlock();
    if (currStmtIdx + 1 < B->size()) {
      CFGElement Next = (*B)[currStmtIdx+1];

      // Is this a constructor for a local variable?
      if (const CFGStmt *StmtElem = dyn_cast<CFGStmt>(&Next)) {
        if (const DeclStmt *DS = dyn_cast<DeclStmt>(StmtElem->getStmt())) {
          if (const VarDecl *Var = dyn_cast<VarDecl>(DS->getSingleDecl())) {
            if (Var->getInit()->IgnoreImplicit() == CE) {
              QualType Ty = Var->getType();
              if (const ArrayType *AT = getContext().getAsArrayType(Ty)) {
                // FIXME: Handle arrays, which run the same constructor for
                // every element. This workaround will just run the first
                // constructor (which should still invalidate the entire array).
                SVal Base = State->getLValue(Var, LCtx);
                Target = State->getLValue(AT->getElementType(),
                                          getSValBuilder().makeZeroArrayIndex(),
                                          Base).getAsRegion();
              } else {
                Target = State->getLValue(Var, LCtx).getAsRegion();
              }
            }
          }
        }
      }
      
      // Is this a constructor for a member?
      if (const CFGInitializer *InitElem = dyn_cast<CFGInitializer>(&Next)) {
        const CXXCtorInitializer *Init = InitElem->getInitializer();
        assert(Init->isAnyMemberInitializer());

        const CXXMethodDecl *CurCtor = cast<CXXMethodDecl>(LCtx->getDecl());
        Loc ThisPtr = getSValBuilder().getCXXThis(CurCtor,
                                                  LCtx->getCurrentStackFrame());
        SVal ThisVal = State->getSVal(ThisPtr);

        if (Init->isIndirectMemberInitializer()) {
          SVal Field = State->getLValue(Init->getIndirectMember(), ThisVal);
          Target = Field.getAsRegion();
        } else {
          SVal Field = State->getLValue(Init->getMember(), ThisVal);
          Target = Field.getAsRegion();
        }
      }

      // FIXME: This will eventually need to handle new-expressions as well.
    }

    // If we couldn't find an existing region to construct into, assume we're
    // constructing a temporary.
    if (!Target) {
      MemRegionManager &MRMgr = getSValBuilder().getRegionManager();
      Target = MRMgr.getCXXTempObjectRegion(CE, LCtx);
    }

    break;
  }
  case CXXConstructExpr::CK_NonVirtualBase:
  case CXXConstructExpr::CK_VirtualBase:
  case CXXConstructExpr::CK_Delegating: {
    const CXXMethodDecl *CurCtor = cast<CXXMethodDecl>(LCtx->getDecl());
    Loc ThisPtr = getSValBuilder().getCXXThis(CurCtor,
                                              LCtx->getCurrentStackFrame());
    SVal ThisVal = State->getSVal(ThisPtr);

    if (CE->getConstructionKind() == CXXConstructExpr::CK_Delegating) {
      Target = ThisVal.getAsRegion();
    } else {
      // Cast to the base type.
      QualType BaseTy = CE->getType();
      SVal BaseVal = getStoreManager().evalDerivedToBase(ThisVal, BaseTy);
      Target = BaseVal.getAsRegion();
    }
    break;
  }
  }

  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<CXXConstructorCall> Call =
    CEMgr.getCXXConstructorCall(CE, Target, State, LCtx);

  ExplodedNodeSet DstPreVisit;
  getCheckerManager().runCheckersForPreStmt(DstPreVisit, Pred, CE, *this);
  ExplodedNodeSet DstPreCall;
  getCheckerManager().runCheckersForPreCall(DstPreCall, DstPreVisit,
                                            *Call, *this);

  ExplodedNodeSet DstInvalidated;
  StmtNodeBuilder Bldr(DstPreCall, DstInvalidated, *currBldrCtx);
  for (ExplodedNodeSet::iterator I = DstPreCall.begin(), E = DstPreCall.end();
       I != E; ++I)
    defaultEvalCall(Bldr, *I, *Call);

  ExplodedNodeSet DstPostCall;
  getCheckerManager().runCheckersForPostCall(DstPostCall, DstInvalidated,
                                             *Call, *this);
  getCheckerManager().runCheckersForPostStmt(destNodes, DstPostCall, CE, *this);
}

void ExprEngine::VisitCXXDestructor(QualType ObjectType,
                                    const MemRegion *Dest,
                                    const Stmt *S,
                                    bool IsBaseDtor,
                                    ExplodedNode *Pred, 
                                    ExplodedNodeSet &Dst) {
  const LocationContext *LCtx = Pred->getLocationContext();
  ProgramStateRef State = Pred->getState();

  // FIXME: We need to run the same destructor on every element of the array.
  // This workaround will just run the first destructor (which will still
  // invalidate the entire array).
  if (const ArrayType *AT = getContext().getAsArrayType(ObjectType)) {
    ObjectType = AT->getElementType();
    Dest = State->getLValue(ObjectType, getSValBuilder().makeZeroArrayIndex(),
                            loc::MemRegionVal(Dest)).getAsRegion();
  }

  const CXXRecordDecl *RecordDecl = ObjectType->getAsCXXRecordDecl();
  assert(RecordDecl && "Only CXXRecordDecls should have destructors");
  const CXXDestructorDecl *DtorDecl = RecordDecl->getDestructor();

  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<CXXDestructorCall> Call =
    CEMgr.getCXXDestructorCall(DtorDecl, S, Dest, IsBaseDtor, State, LCtx);

  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                Call->getSourceRange().getBegin(),
                                "Error evaluating destructor");

  ExplodedNodeSet DstPreCall;
  getCheckerManager().runCheckersForPreCall(DstPreCall, Pred,
                                            *Call, *this);

  ExplodedNodeSet DstInvalidated;
  StmtNodeBuilder Bldr(DstPreCall, DstInvalidated, *currBldrCtx);
  for (ExplodedNodeSet::iterator I = DstPreCall.begin(), E = DstPreCall.end();
       I != E; ++I)
    defaultEvalCall(Bldr, *I, *Call);

  ExplodedNodeSet DstPostCall;
  getCheckerManager().runCheckersForPostCall(Dst, DstInvalidated,
                                             *Call, *this);
}

void ExprEngine::VisitCXXNewExpr(const CXXNewExpr *CNE, ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  // FIXME: Much of this should eventually migrate to CXXAllocatorCall.
  // Also, we need to decide how allocators actually work -- they're not
  // really part of the CXXNewExpr because they happen BEFORE the
  // CXXConstructExpr subexpression. See PR12014 for some discussion.
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  
  unsigned blockCount = currBldrCtx->blockCount();
  const LocationContext *LCtx = Pred->getLocationContext();
  DefinedOrUnknownSVal symVal = svalBuilder.conjureSymbolVal(0, CNE, LCtx,
                                                             CNE->getType(),
                                                             blockCount);
  ProgramStateRef State = Pred->getState();

  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<CXXAllocatorCall> Call =
    CEMgr.getCXXAllocatorCall(CNE, State, LCtx);

  // Invalidate placement args.
  // FIXME: Once we figure out how we want allocators to work,
  // we should be using the usual pre-/(default-)eval-/post-call checks here.
  State = Call->invalidateRegions(blockCount);

  // If we're compiling with exceptions enabled, and this allocation function
  // is not declared as non-throwing, failures /must/ be signalled by
  // exceptions, and thus the return value will never be NULL.
  // C++11 [basic.stc.dynamic.allocation]p3.
  FunctionDecl *FD = CNE->getOperatorNew();
  if (FD && getContext().getLangOpts().CXXExceptions) {
    QualType Ty = FD->getType();
    if (const FunctionProtoType *ProtoType = Ty->getAs<FunctionProtoType>())
      if (!ProtoType->isNothrow(getContext()))
        State = State->assume(symVal, true);
  }

  if (CNE->isArray()) {
    // FIXME: allocating an array requires simulating the constructors.
    // For now, just return a symbolicated region.
    const MemRegion *NewReg = cast<loc::MemRegionVal>(symVal).getRegion();
    QualType ObjTy = CNE->getType()->getAs<PointerType>()->getPointeeType();
    const ElementRegion *EleReg =
      getStoreManager().GetElementZeroRegion(NewReg, ObjTy);
    State = State->BindExpr(CNE, Pred->getLocationContext(),
                            loc::MemRegionVal(EleReg));
    Bldr.generateNode(CNE, Pred, State);
    return;
  }

  // FIXME: Once we have proper support for CXXConstructExprs inside
  // CXXNewExpr, we need to make sure that the constructed object is not
  // immediately invalidated here. (The placement call should happen before
  // the constructor call anyway.)
  if (FD && FD->isReservedGlobalPlacementOperator()) {
    // Non-array placement new should always return the placement location.
    SVal PlacementLoc = State->getSVal(CNE->getPlacementArg(0), LCtx);
    SVal Result = svalBuilder.evalCast(PlacementLoc, CNE->getType(),
                                       CNE->getPlacementArg(0)->getType());
    State = State->BindExpr(CNE, LCtx, Result);
  } else {
    State = State->BindExpr(CNE, LCtx, symVal);
  }

  // If the type is not a record, we won't have a CXXConstructExpr as an
  // initializer. Copy the value over.
  if (const Expr *Init = CNE->getInitializer()) {
    if (!isa<CXXConstructExpr>(Init)) {
      QualType ObjTy = CNE->getType()->getAs<PointerType>()->getPointeeType();
      (void)ObjTy;
      assert(!ObjTy->isRecordType());
      SVal Location = State->getSVal(CNE, LCtx);
      if (isa<Loc>(Location))
        State = State->bindLoc(cast<Loc>(Location), State->getSVal(Init, LCtx));
    }
  }

  Bldr.generateNode(CNE, Pred, State);
}

void ExprEngine::VisitCXXDeleteExpr(const CXXDeleteExpr *CDE, 
                                    ExplodedNode *Pred, ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  ProgramStateRef state = Pred->getState();
  Bldr.generateNode(CDE, Pred, state);
}

void ExprEngine::VisitCXXCatchStmt(const CXXCatchStmt *CS,
                                   ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  const VarDecl *VD = CS->getExceptionDecl();
  if (!VD) {
    Dst.Add(Pred);
    return;
  }

  const LocationContext *LCtx = Pred->getLocationContext();
  SVal V = svalBuilder.conjureSymbolVal(CS, LCtx, VD->getType(),
                                        currBldrCtx->blockCount());
  ProgramStateRef state = Pred->getState();
  state = state->bindLoc(state->getLValue(VD, LCtx), V);

  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  Bldr.generateNode(CS, Pred, state);
}

void ExprEngine::VisitCXXThisExpr(const CXXThisExpr *TE, ExplodedNode *Pred,
                                    ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);

  // Get the this object region from StoreManager.
  const LocationContext *LCtx = Pred->getLocationContext();
  const MemRegion *R =
    svalBuilder.getRegionManager().getCXXThisRegion(
                                  getContext().getCanonicalType(TE->getType()),
                                                    LCtx);

  ProgramStateRef state = Pred->getState();
  SVal V = state->getSVal(loc::MemRegionVal(R));
  Bldr.generateNode(TE, Pred, state->BindExpr(TE, LCtx, V));
}
