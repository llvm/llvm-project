//=-- ExprEngineC.cpp - ExprEngine support for C expressions ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines ExprEngine's support for C expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"

using namespace clang;
using namespace ento;
using llvm::APSInt;

void ExprEngine::VisitBinaryOperator(const BinaryOperator* B,
                                     ExplodedNode *Pred,
                                     ExplodedNodeSet &Dst) {

  Expr *LHS = B->getLHS()->IgnoreParens();
  Expr *RHS = B->getRHS()->IgnoreParens();
  
  // FIXME: Prechecks eventually go in ::Visit().
  ExplodedNodeSet CheckedSet;
  ExplodedNodeSet Tmp2;
  getCheckerManager().runCheckersForPreStmt(CheckedSet, Pred, B, *this);
    
  // With both the LHS and RHS evaluated, process the operation itself.    
  for (ExplodedNodeSet::iterator it=CheckedSet.begin(), ei=CheckedSet.end();
         it != ei; ++it) {
      
    ProgramStateRef state = (*it)->getState();
    const LocationContext *LCtx = (*it)->getLocationContext();
    SVal LeftV = state->getSVal(LHS, LCtx);
    SVal RightV = state->getSVal(RHS, LCtx);
      
    BinaryOperator::Opcode Op = B->getOpcode();
      
    if (Op == BO_Assign) {
      // EXPERIMENTAL: "Conjured" symbols.
      // FIXME: Handle structs.
      if (RightV.isUnknown()) {
        unsigned Count = currBldrCtx->blockCount();
        RightV = svalBuilder.conjureSymbolVal(0, B->getRHS(), LCtx, Count);
      }
      // Simulate the effects of a "store":  bind the value of the RHS
      // to the L-Value represented by the LHS.
      SVal ExprVal = B->isGLValue() ? LeftV : RightV;
      evalStore(Tmp2, B, LHS, *it, state->BindExpr(B, LCtx, ExprVal),
                LeftV, RightV);
      continue;
    }
      
    if (!B->isAssignmentOp()) {
      StmtNodeBuilder Bldr(*it, Tmp2, *currBldrCtx);

      if (B->isAdditiveOp()) {
        // If one of the operands is a location, conjure a symbol for the other
        // one (offset) if it's unknown so that memory arithmetic always
        // results in an ElementRegion.
        // TODO: This can be removed after we enable history tracking with
        // SymSymExpr.
        unsigned Count = currBldrCtx->blockCount();
        if (isa<Loc>(LeftV) &&
            RHS->getType()->isIntegerType() && RightV.isUnknown()) {
          RightV = svalBuilder.conjureSymbolVal(RHS, LCtx, RHS->getType(),
                                                Count);
        }
        if (isa<Loc>(RightV) &&
            LHS->getType()->isIntegerType() && LeftV.isUnknown()) {
          LeftV = svalBuilder.conjureSymbolVal(LHS, LCtx, LHS->getType(),
                                               Count);
        }
      }

      // Process non-assignments except commas or short-circuited
      // logical expressions (LAnd and LOr).
      SVal Result = evalBinOp(state, Op, LeftV, RightV, B->getType());      
      if (Result.isUnknown()) {
        Bldr.generateNode(B, *it, state);
        continue;
      }        

      state = state->BindExpr(B, LCtx, Result);      
      Bldr.generateNode(B, *it, state);
      continue;
    }
      
    assert (B->isCompoundAssignmentOp());
    
    switch (Op) {
      default:
        llvm_unreachable("Invalid opcode for compound assignment.");
      case BO_MulAssign: Op = BO_Mul; break;
      case BO_DivAssign: Op = BO_Div; break;
      case BO_RemAssign: Op = BO_Rem; break;
      case BO_AddAssign: Op = BO_Add; break;
      case BO_SubAssign: Op = BO_Sub; break;
      case BO_ShlAssign: Op = BO_Shl; break;
      case BO_ShrAssign: Op = BO_Shr; break;
      case BO_AndAssign: Op = BO_And; break;
      case BO_XorAssign: Op = BO_Xor; break;
      case BO_OrAssign:  Op = BO_Or;  break;
    }
      
    // Perform a load (the LHS).  This performs the checks for
    // null dereferences, and so on.
    ExplodedNodeSet Tmp;
    SVal location = LeftV;
    evalLoad(Tmp, B, LHS, *it, state, location);
    
    for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I != E;
         ++I) {

      state = (*I)->getState();
      const LocationContext *LCtx = (*I)->getLocationContext();
      SVal V = state->getSVal(LHS, LCtx);
      
      // Get the computation type.
      QualType CTy =
        cast<CompoundAssignOperator>(B)->getComputationResultType();
      CTy = getContext().getCanonicalType(CTy);
      
      QualType CLHSTy =
        cast<CompoundAssignOperator>(B)->getComputationLHSType();
      CLHSTy = getContext().getCanonicalType(CLHSTy);
      
      QualType LTy = getContext().getCanonicalType(LHS->getType());
      
      // Promote LHS.
      V = svalBuilder.evalCast(V, CLHSTy, LTy);
      
      // Compute the result of the operation.
      SVal Result = svalBuilder.evalCast(evalBinOp(state, Op, V, RightV, CTy),
                                         B->getType(), CTy);
      
      // EXPERIMENTAL: "Conjured" symbols.
      // FIXME: Handle structs.
      
      SVal LHSVal;
      
      if (Result.isUnknown()) {
        // The symbolic value is actually for the type of the left-hand side
        // expression, not the computation type, as this is the value the
        // LValue on the LHS will bind to.
        LHSVal = svalBuilder.conjureSymbolVal(0, B->getRHS(), LCtx, LTy,
                                              currBldrCtx->blockCount());
        // However, we need to convert the symbol to the computation type.
        Result = svalBuilder.evalCast(LHSVal, CTy, LTy);
      }
      else {
        // The left-hand side may bind to a different value then the
        // computation type.
        LHSVal = svalBuilder.evalCast(Result, LTy, CTy);
      }
      
      // In C++, assignment and compound assignment operators return an 
      // lvalue.
      if (B->isGLValue())
        state = state->BindExpr(B, LCtx, location);
      else
        state = state->BindExpr(B, LCtx, Result);
      
      evalStore(Tmp2, B, LHS, *I, state, location, LHSVal);
    }
  }
  
  // FIXME: postvisits eventually go in ::Visit()
  getCheckerManager().runCheckersForPostStmt(Dst, Tmp2, B, *this);
}

void ExprEngine::VisitBlockExpr(const BlockExpr *BE, ExplodedNode *Pred,
                                ExplodedNodeSet &Dst) {
  
  CanQualType T = getContext().getCanonicalType(BE->getType());

  // Get the value of the block itself.
  SVal V = svalBuilder.getBlockPointer(BE->getBlockDecl(), T,
                                       Pred->getLocationContext());
  
  ProgramStateRef State = Pred->getState();
  
  // If we created a new MemRegion for the block, we should explicitly bind
  // the captured variables.
  if (const BlockDataRegion *BDR =
      dyn_cast_or_null<BlockDataRegion>(V.getAsRegion())) {
    
    BlockDataRegion::referenced_vars_iterator I = BDR->referenced_vars_begin(),
                                              E = BDR->referenced_vars_end();
    
    for (; I != E; ++I) {
      const MemRegion *capturedR = I.getCapturedRegion();
      const MemRegion *originalR = I.getOriginalRegion();
      if (capturedR != originalR) {
        SVal originalV = State->getSVal(loc::MemRegionVal(originalR));
        State = State->bindLoc(loc::MemRegionVal(capturedR), originalV);
      }
    }
  }
  
  ExplodedNodeSet Tmp;
  StmtNodeBuilder Bldr(Pred, Tmp, *currBldrCtx);
  Bldr.generateNode(BE, Pred,
                    State->BindExpr(BE, Pred->getLocationContext(), V),
                    0, ProgramPoint::PostLValueKind);
  
  // FIXME: Move all post/pre visits to ::Visit().
  getCheckerManager().runCheckersForPostStmt(Dst, Tmp, BE, *this);
}

void ExprEngine::VisitCast(const CastExpr *CastE, const Expr *Ex, 
                           ExplodedNode *Pred, ExplodedNodeSet &Dst) {
  
  ExplodedNodeSet dstPreStmt;
  getCheckerManager().runCheckersForPreStmt(dstPreStmt, Pred, CastE, *this);
  
  if (CastE->getCastKind() == CK_LValueToRValue) {
    for (ExplodedNodeSet::iterator I = dstPreStmt.begin(), E = dstPreStmt.end();
         I!=E; ++I) {
      ExplodedNode *subExprNode = *I;
      ProgramStateRef state = subExprNode->getState();
      const LocationContext *LCtx = subExprNode->getLocationContext();
      evalLoad(Dst, CastE, CastE, subExprNode, state, state->getSVal(Ex, LCtx));
    }
    return;
  }
  
  // All other casts.  
  QualType T = CastE->getType();
  QualType ExTy = Ex->getType();
  
  if (const ExplicitCastExpr *ExCast=dyn_cast_or_null<ExplicitCastExpr>(CastE))
    T = ExCast->getTypeAsWritten();
  
  StmtNodeBuilder Bldr(dstPreStmt, Dst, *currBldrCtx);
  for (ExplodedNodeSet::iterator I = dstPreStmt.begin(), E = dstPreStmt.end();
       I != E; ++I) {
    
    Pred = *I;
    ProgramStateRef state = Pred->getState();
    const LocationContext *LCtx = Pred->getLocationContext();

    switch (CastE->getCastKind()) {
      case CK_LValueToRValue:
        llvm_unreachable("LValueToRValue casts handled earlier.");
      case CK_ToVoid:
        continue;
        // The analyzer doesn't do anything special with these casts,
        // since it understands retain/release semantics already.
      case CK_ARCProduceObject:
      case CK_ARCConsumeObject:
      case CK_ARCReclaimReturnedObject:
      case CK_ARCExtendBlockObject: // Fall-through.
      case CK_CopyAndAutoreleaseBlockObject:
        // The analyser can ignore atomic casts for now, although some future
        // checkers may want to make certain that you're not modifying the same
        // value through atomic and nonatomic pointers.
      case CK_AtomicToNonAtomic:
      case CK_NonAtomicToAtomic:
        // True no-ops.
      case CK_NoOp:
      case CK_ConstructorConversion:
      case CK_UserDefinedConversion:
      case CK_FunctionToPointerDecay:
      case CK_BuiltinFnToFnPtr: {
        // Copy the SVal of Ex to CastE.
        ProgramStateRef state = Pred->getState();
        const LocationContext *LCtx = Pred->getLocationContext();
        SVal V = state->getSVal(Ex, LCtx);
        state = state->BindExpr(CastE, LCtx, V);
        Bldr.generateNode(CastE, Pred, state);
        continue;
      }
      case CK_MemberPointerToBoolean:
        // FIXME: For now, member pointers are represented by void *.
        // FALLTHROUGH
      case CK_Dependent:
      case CK_ArrayToPointerDecay:
      case CK_BitCast:
      case CK_IntegralCast:
      case CK_NullToPointer:
      case CK_IntegralToPointer:
      case CK_PointerToIntegral:
      case CK_PointerToBoolean:
      case CK_IntegralToBoolean:
      case CK_IntegralToFloating:
      case CK_FloatingToIntegral:
      case CK_FloatingToBoolean:
      case CK_FloatingCast:
      case CK_FloatingRealToComplex:
      case CK_FloatingComplexToReal:
      case CK_FloatingComplexToBoolean:
      case CK_FloatingComplexCast:
      case CK_FloatingComplexToIntegralComplex:
      case CK_IntegralRealToComplex:
      case CK_IntegralComplexToReal:
      case CK_IntegralComplexToBoolean:
      case CK_IntegralComplexCast:
      case CK_IntegralComplexToFloatingComplex:
      case CK_CPointerToObjCPointerCast:
      case CK_BlockPointerToObjCPointerCast:
      case CK_AnyPointerToBlockPointerCast:  
      case CK_ObjCObjectLValueCast: {
        // Delegate to SValBuilder to process.
        SVal V = state->getSVal(Ex, LCtx);
        V = svalBuilder.evalCast(V, T, ExTy);
        state = state->BindExpr(CastE, LCtx, V);
        Bldr.generateNode(CastE, Pred, state);
        continue;
      }
      case CK_DerivedToBase:
      case CK_UncheckedDerivedToBase: {
        // For DerivedToBase cast, delegate to the store manager.
        SVal val = state->getSVal(Ex, LCtx);
        val = getStoreManager().evalDerivedToBase(val, CastE);
        state = state->BindExpr(CastE, LCtx, val);
        Bldr.generateNode(CastE, Pred, state);
        continue;
      }
      // Handle C++ dyn_cast.
      case CK_Dynamic: {
        SVal val = state->getSVal(Ex, LCtx);

        // Compute the type of the result.
        QualType resultType = CastE->getType();
        if (CastE->isGLValue())
          resultType = getContext().getPointerType(resultType);

        bool Failed = false;

        // Check if the value being cast evaluates to 0.
        if (val.isZeroConstant())
          Failed = true;
        // Else, evaluate the cast.
        else
          val = getStoreManager().evalDynamicCast(val, T, Failed);

        if (Failed) {
          if (T->isReferenceType()) {
            // A bad_cast exception is thrown if input value is a reference.
            // Currently, we model this, by generating a sink.
            Bldr.generateSink(CastE, Pred, state);
            continue;
          } else {
            // If the cast fails on a pointer, bind to 0.
            state = state->BindExpr(CastE, LCtx, svalBuilder.makeNull());
          }
        } else {
          // If we don't know if the cast succeeded, conjure a new symbol.
          if (val.isUnknown()) {
            DefinedOrUnknownSVal NewSym =
              svalBuilder.conjureSymbolVal(0, CastE, LCtx, resultType,
                                           currBldrCtx->blockCount());
            state = state->BindExpr(CastE, LCtx, NewSym);
          } else 
            // Else, bind to the derived region value.
            state = state->BindExpr(CastE, LCtx, val);
        }
        Bldr.generateNode(CastE, Pred, state);
        continue;
      }
      case CK_NullToMemberPointer: {
        // FIXME: For now, member pointers are represented by void *.
        SVal V = svalBuilder.makeIntValWithPtrWidth(0, true);
        state = state->BindExpr(CastE, LCtx, V);
        Bldr.generateNode(CastE, Pred, state);
        continue;
      }
      // Various C++ casts that are not handled yet.
      case CK_ToUnion:
      case CK_BaseToDerived:
      case CK_BaseToDerivedMemberPointer:
      case CK_DerivedToBaseMemberPointer:
      case CK_ReinterpretMemberPointer:
      case CK_VectorSplat:
      case CK_LValueBitCast: {
        // Recover some path-sensitivty by conjuring a new value.
        QualType resultType = CastE->getType();
        if (CastE->isGLValue())
          resultType = getContext().getPointerType(resultType);
        SVal result = svalBuilder.conjureSymbolVal(0, CastE, LCtx,
                                                   resultType,
                                                   currBldrCtx->blockCount());
        state = state->BindExpr(CastE, LCtx, result);
        Bldr.generateNode(CastE, Pred, state);
        continue;
      }
    }
  }
}

void ExprEngine::VisitCompoundLiteralExpr(const CompoundLiteralExpr *CL,
                                          ExplodedNode *Pred,
                                          ExplodedNodeSet &Dst) {
  StmtNodeBuilder B(Pred, Dst, *currBldrCtx);

  const InitListExpr *ILE 
    = cast<InitListExpr>(CL->getInitializer()->IgnoreParens());
  
  ProgramStateRef state = Pred->getState();
  SVal ILV = state->getSVal(ILE, Pred->getLocationContext());
  const LocationContext *LC = Pred->getLocationContext();
  state = state->bindCompoundLiteral(CL, LC, ILV);

  // Compound literal expressions are a GNU extension in C++.
  // Unlike in C, where CLs are lvalues, in C++ CLs are prvalues,
  // and like temporary objects created by the functional notation T()
  // CLs are destroyed at the end of the containing full-expression.
  // HOWEVER, an rvalue of array type is not something the analyzer can
  // reason about, since we expect all regions to be wrapped in Locs.
  // So we treat array CLs as lvalues as well, knowing that they will decay
  // to pointers as soon as they are used.
  if (CL->isGLValue() || CL->getType()->isArrayType())
    B.generateNode(CL, Pred, state->BindExpr(CL, LC, state->getLValue(CL, LC)));
  else
    B.generateNode(CL, Pred, state->BindExpr(CL, LC, ILV));
}

void ExprEngine::VisitDeclStmt(const DeclStmt *DS, ExplodedNode *Pred,
                               ExplodedNodeSet &Dst) {
  
  // FIXME: static variables may have an initializer, but the second
  //  time a function is called those values may not be current.
  //  This may need to be reflected in the CFG.
  
  // Assumption: The CFG has one DeclStmt per Decl.
  const Decl *D = *DS->decl_begin();
  
  if (!D || !isa<VarDecl>(D)) {
    //TODO:AZ: remove explicit insertion after refactoring is done.
    Dst.insert(Pred);
    return;
  }
  
  // FIXME: all pre/post visits should eventually be handled by ::Visit().
  ExplodedNodeSet dstPreVisit;
  getCheckerManager().runCheckersForPreStmt(dstPreVisit, Pred, DS, *this);
  
  StmtNodeBuilder B(dstPreVisit, Dst, *currBldrCtx);
  const VarDecl *VD = dyn_cast<VarDecl>(D);
  for (ExplodedNodeSet::iterator I = dstPreVisit.begin(), E = dstPreVisit.end();
       I!=E; ++I) {
    ExplodedNode *N = *I;
    ProgramStateRef state = N->getState();
    
    // Decls without InitExpr are not initialized explicitly.
    const LocationContext *LC = N->getLocationContext();
    
    if (const Expr *InitEx = VD->getInit()) {
      SVal InitVal = state->getSVal(InitEx, LC);

      if (InitVal == state->getLValue(VD, LC) ||
          (VD->getType()->isArrayType() &&
           isa<CXXConstructExpr>(InitEx->IgnoreImplicit()))) {
        // We constructed the object directly in the variable.
        // No need to bind anything.
        B.generateNode(DS, N, state);
      } else {
        // We bound the temp obj region to the CXXConstructExpr. Now recover
        // the lazy compound value when the variable is not a reference.
        if (AMgr.getLangOpts().CPlusPlus && VD->getType()->isRecordType() && 
            !VD->getType()->isReferenceType() && isa<loc::MemRegionVal>(InitVal)){
          InitVal = state->getSVal(cast<loc::MemRegionVal>(InitVal).getRegion());
          assert(isa<nonloc::LazyCompoundVal>(InitVal));
        }
        
        // Recover some path-sensitivity if a scalar value evaluated to
        // UnknownVal.
        if (InitVal.isUnknown()) {
          QualType Ty = InitEx->getType();
          if (InitEx->isGLValue()) {
            Ty = getContext().getPointerType(Ty);
          }

          InitVal = svalBuilder.conjureSymbolVal(0, InitEx, LC, Ty,
                                                 currBldrCtx->blockCount());
        }
        B.takeNodes(N);
        ExplodedNodeSet Dst2;
        evalBind(Dst2, DS, N, state->getLValue(VD, LC), InitVal, true);
        B.addNodes(Dst2);
      }
    }
    else {
      B.generateNode(DS, N, state);
    }
  }
}

void ExprEngine::VisitLogicalExpr(const BinaryOperator* B, ExplodedNode *Pred,
                                  ExplodedNodeSet &Dst) {
  assert(B->getOpcode() == BO_LAnd ||
         B->getOpcode() == BO_LOr);

  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  ProgramStateRef state = Pred->getState();

  ExplodedNode *N = Pred;
  while (!isa<BlockEntrance>(N->getLocation())) {
    ProgramPoint P = N->getLocation();
    assert(isa<PreStmt>(P)|| isa<PreStmtPurgeDeadSymbols>(P));
    (void) P;
    assert(N->pred_size() == 1);
    N = *N->pred_begin();
  }
  assert(N->pred_size() == 1);
  N = *N->pred_begin();
  BlockEdge BE = cast<BlockEdge>(N->getLocation());
  SVal X;

  // Determine the value of the expression by introspecting how we
  // got this location in the CFG.  This requires looking at the previous
  // block we were in and what kind of control-flow transfer was involved.
  const CFGBlock *SrcBlock = BE.getSrc();
  // The only terminator (if there is one) that makes sense is a logical op.
  CFGTerminator T = SrcBlock->getTerminator();
  if (const BinaryOperator *Term = cast_or_null<BinaryOperator>(T.getStmt())) {
    (void) Term;
    assert(Term->isLogicalOp());
    assert(SrcBlock->succ_size() == 2);
    // Did we take the true or false branch?
    unsigned constant = (*SrcBlock->succ_begin() == BE.getDst()) ? 1 : 0;
    X = svalBuilder.makeIntVal(constant, B->getType());
  }
  else {
    // If there is no terminator, by construction the last statement
    // in SrcBlock is the value of the enclosing expression.
    // However, we still need to constrain that value to be 0 or 1.
    assert(!SrcBlock->empty());
    CFGStmt Elem = cast<CFGStmt>(*SrcBlock->rbegin());
    const Expr *RHS = cast<Expr>(Elem.getStmt());
    SVal RHSVal = N->getState()->getSVal(RHS, Pred->getLocationContext());

    DefinedOrUnknownSVal DefinedRHS = cast<DefinedOrUnknownSVal>(RHSVal);
    ProgramStateRef StTrue, StFalse;
    llvm::tie(StTrue, StFalse) = N->getState()->assume(DefinedRHS);
    if (StTrue) {
      if (StFalse) {
        // We can't constrain the value to 0 or 1; the best we can do is a cast.
        X = getSValBuilder().evalCast(RHSVal, B->getType(), RHS->getType());
      } else {
        // The value is known to be true.
        X = getSValBuilder().makeIntVal(1, B->getType());
      }
    } else {
      // The value is known to be false.
      assert(StFalse && "Infeasible path!");
      X = getSValBuilder().makeIntVal(0, B->getType());
    }
  }

  Bldr.generateNode(B, Pred, state->BindExpr(B, Pred->getLocationContext(), X));
}

void ExprEngine::VisitInitListExpr(const InitListExpr *IE,
                                   ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  StmtNodeBuilder B(Pred, Dst, *currBldrCtx);

  ProgramStateRef state = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();
  QualType T = getContext().getCanonicalType(IE->getType());
  unsigned NumInitElements = IE->getNumInits();
  
  if (T->isArrayType() || T->isRecordType() || T->isVectorType() ||
      T->isAnyComplexType()) {
    llvm::ImmutableList<SVal> vals = getBasicVals().getEmptySValList();
    
    // Handle base case where the initializer has no elements.
    // e.g: static int* myArray[] = {};
    if (NumInitElements == 0) {
      SVal V = svalBuilder.makeCompoundVal(T, vals);
      B.generateNode(IE, Pred, state->BindExpr(IE, LCtx, V));
      return;
    }
    
    for (InitListExpr::const_reverse_iterator it = IE->rbegin(),
         ei = IE->rend(); it != ei; ++it) {
      vals = getBasicVals().consVals(state->getSVal(cast<Expr>(*it), LCtx),
                                     vals);
    }
    
    B.generateNode(IE, Pred,
                   state->BindExpr(IE, LCtx,
                                   svalBuilder.makeCompoundVal(T, vals)));
    return;
  }

  // Handle scalars: int{5} and int{}.
  assert(NumInitElements <= 1);

  SVal V;
  if (NumInitElements == 0)
    V = getSValBuilder().makeZeroVal(T);
  else
    V = state->getSVal(IE->getInit(0), LCtx);

  B.generateNode(IE, Pred, state->BindExpr(IE, LCtx, V));
}

void ExprEngine::VisitGuardedExpr(const Expr *Ex,
                                  const Expr *L, 
                                  const Expr *R,
                                  ExplodedNode *Pred,
                                  ExplodedNodeSet &Dst) {
  StmtNodeBuilder B(Pred, Dst, *currBldrCtx);
  ProgramStateRef state = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();
  const CFGBlock *SrcBlock = 0;

  for (const ExplodedNode *N = Pred ; N ; N = *N->pred_begin()) {
    ProgramPoint PP = N->getLocation();
    if (isa<PreStmtPurgeDeadSymbols>(PP) || isa<BlockEntrance>(PP)) {
      assert(N->pred_size() == 1);
      continue;
    }
    SrcBlock = cast<BlockEdge>(&PP)->getSrc();
    break;
  }

  // Find the last expression in the predecessor block.  That is the
  // expression that is used for the value of the ternary expression.
  bool hasValue = false;
  SVal V;

  for (CFGBlock::const_reverse_iterator I = SrcBlock->rbegin(),
                                        E = SrcBlock->rend(); I != E; ++I) {
    CFGElement CE = *I;
    if (CFGStmt *CS = dyn_cast<CFGStmt>(&CE)) {
      const Expr *ValEx = cast<Expr>(CS->getStmt());
      hasValue = true;
      V = state->getSVal(ValEx, LCtx);
      break;
    }
  }

  assert(hasValue);
  (void) hasValue;

  // Generate a new node with the binding from the appropriate path.
  B.generateNode(Ex, Pred, state->BindExpr(Ex, LCtx, V, true));
}

void ExprEngine::
VisitOffsetOfExpr(const OffsetOfExpr *OOE, 
                  ExplodedNode *Pred, ExplodedNodeSet &Dst) {
  StmtNodeBuilder B(Pred, Dst, *currBldrCtx);
  APSInt IV;
  if (OOE->EvaluateAsInt(IV, getContext())) {
    assert(IV.getBitWidth() == getContext().getTypeSize(OOE->getType()));
    assert(OOE->getType()->isIntegerType());
    assert(IV.isSigned() == OOE->getType()->isSignedIntegerOrEnumerationType());
    SVal X = svalBuilder.makeIntVal(IV);
    B.generateNode(OOE, Pred,
                   Pred->getState()->BindExpr(OOE, Pred->getLocationContext(),
                                              X));
  }
  // FIXME: Handle the case where __builtin_offsetof is not a constant.
}


void ExprEngine::
VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *Ex,
                              ExplodedNode *Pred,
                              ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);

  QualType T = Ex->getTypeOfArgument();
  
  if (Ex->getKind() == UETT_SizeOf) {
    if (!T->isIncompleteType() && !T->isConstantSizeType()) {
      assert(T->isVariableArrayType() && "Unknown non-constant-sized type.");
      
      // FIXME: Add support for VLA type arguments and VLA expressions.
      // When that happens, we should probably refactor VLASizeChecker's code.
      return;
    }
    else if (T->getAs<ObjCObjectType>()) {
      // Some code tries to take the sizeof an ObjCObjectType, relying that
      // the compiler has laid out its representation.  Just report Unknown
      // for these.
      return;
    }
  }
  
  APSInt Value = Ex->EvaluateKnownConstInt(getContext());
  CharUnits amt = CharUnits::fromQuantity(Value.getZExtValue());
  
  ProgramStateRef state = Pred->getState();
  state = state->BindExpr(Ex, Pred->getLocationContext(),
                          svalBuilder.makeIntVal(amt.getQuantity(),
                                                     Ex->getType()));
  Bldr.generateNode(Ex, Pred, state);
}

void ExprEngine::VisitUnaryOperator(const UnaryOperator* U, 
                                    ExplodedNode *Pred,
                                    ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  switch (U->getOpcode()) {
    default: {
      Bldr.takeNodes(Pred);
      ExplodedNodeSet Tmp;
      VisitIncrementDecrementOperator(U, Pred, Tmp);
      Bldr.addNodes(Tmp);
    }
      break;
    case UO_Real: {
      const Expr *Ex = U->getSubExpr()->IgnoreParens();
        
      // FIXME: We don't have complex SValues yet.
      if (Ex->getType()->isAnyComplexType()) {
        // Just report "Unknown."
        break;
      }
        
      // For all other types, UO_Real is an identity operation.
      assert (U->getType() == Ex->getType());
      ProgramStateRef state = Pred->getState();
      const LocationContext *LCtx = Pred->getLocationContext();
      Bldr.generateNode(U, Pred, state->BindExpr(U, LCtx,
                                                 state->getSVal(Ex, LCtx)));
      break;
    }
      
    case UO_Imag: {      
      const Expr *Ex = U->getSubExpr()->IgnoreParens();
      // FIXME: We don't have complex SValues yet.
      if (Ex->getType()->isAnyComplexType()) {
        // Just report "Unknown."
        break;
      }
      // For all other types, UO_Imag returns 0.
      ProgramStateRef state = Pred->getState();
      const LocationContext *LCtx = Pred->getLocationContext();
      SVal X = svalBuilder.makeZeroVal(Ex->getType());
      Bldr.generateNode(U, Pred, state->BindExpr(U, LCtx, X));
      break;
    }
      
    case UO_Plus:
      assert(!U->isGLValue());
      // FALL-THROUGH.
    case UO_Deref:
    case UO_AddrOf:
    case UO_Extension: {
      // FIXME: We can probably just have some magic in Environment::getSVal()
      // that propagates values, instead of creating a new node here.
      //
      // Unary "+" is a no-op, similar to a parentheses.  We still have places
      // where it may be a block-level expression, so we need to
      // generate an extra node that just propagates the value of the
      // subexpression.      
      const Expr *Ex = U->getSubExpr()->IgnoreParens();
      ProgramStateRef state = Pred->getState();
      const LocationContext *LCtx = Pred->getLocationContext();
      Bldr.generateNode(U, Pred, state->BindExpr(U, LCtx,
                                                 state->getSVal(Ex, LCtx)));
      break;
    }
      
    case UO_LNot:
    case UO_Minus:
    case UO_Not: {
      assert (!U->isGLValue());
      const Expr *Ex = U->getSubExpr()->IgnoreParens();
      ProgramStateRef state = Pred->getState();
      const LocationContext *LCtx = Pred->getLocationContext();
        
      // Get the value of the subexpression.
      SVal V = state->getSVal(Ex, LCtx);
        
      if (V.isUnknownOrUndef()) {
        Bldr.generateNode(U, Pred, state->BindExpr(U, LCtx, V));
        break;
      }
        
      switch (U->getOpcode()) {
        default:
          llvm_unreachable("Invalid Opcode.");
        case UO_Not:
          // FIXME: Do we need to handle promotions?
          state = state->BindExpr(U, LCtx, evalComplement(cast<NonLoc>(V)));
          break;
        case UO_Minus:
          // FIXME: Do we need to handle promotions?
          state = state->BindExpr(U, LCtx, evalMinus(cast<NonLoc>(V)));
          break;
        case UO_LNot:
          // C99 6.5.3.3: "The expression !E is equivalent to (0==E)."
          //
          //  Note: technically we do "E == 0", but this is the same in the
          //    transfer functions as "0 == E".
          SVal Result;          
          if (isa<Loc>(V)) {
            Loc X = svalBuilder.makeNull();
            Result = evalBinOp(state, BO_EQ, cast<Loc>(V), X,
                               U->getType());
          }
          else {
            nonloc::ConcreteInt X(getBasicVals().getValue(0, Ex->getType()));
            Result = evalBinOp(state, BO_EQ, cast<NonLoc>(V), X,
                               U->getType());
          }
          
          state = state->BindExpr(U, LCtx, Result);          
          break;
      }
      Bldr.generateNode(U, Pred, state);
      break;
    }
  }

}

void ExprEngine::VisitIncrementDecrementOperator(const UnaryOperator* U,
                                                 ExplodedNode *Pred,
                                                 ExplodedNodeSet &Dst) {
  // Handle ++ and -- (both pre- and post-increment).
  assert (U->isIncrementDecrementOp());
  const Expr *Ex = U->getSubExpr()->IgnoreParens();
  
  const LocationContext *LCtx = Pred->getLocationContext();
  ProgramStateRef state = Pred->getState();
  SVal loc = state->getSVal(Ex, LCtx);
  
  // Perform a load.
  ExplodedNodeSet Tmp;
  evalLoad(Tmp, U, Ex, Pred, state, loc);
  
  ExplodedNodeSet Dst2;
  StmtNodeBuilder Bldr(Tmp, Dst2, *currBldrCtx);
  for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end();I!=E;++I) {
    
    state = (*I)->getState();
    assert(LCtx == (*I)->getLocationContext());
    SVal V2_untested = state->getSVal(Ex, LCtx);
    
    // Propagate unknown and undefined values.
    if (V2_untested.isUnknownOrUndef()) {
      Bldr.generateNode(U, *I, state->BindExpr(U, LCtx, V2_untested));
      continue;
    }
    DefinedSVal V2 = cast<DefinedSVal>(V2_untested);
    
    // Handle all other values.
    BinaryOperator::Opcode Op = U->isIncrementOp() ? BO_Add : BO_Sub;
    
    // If the UnaryOperator has non-location type, use its type to create the
    // constant value. If the UnaryOperator has location type, create the
    // constant with int type and pointer width.
    SVal RHS;
    
    if (U->getType()->isAnyPointerType())
      RHS = svalBuilder.makeArrayIndex(1);
    else if (U->getType()->isIntegralOrEnumerationType())
      RHS = svalBuilder.makeIntVal(1, U->getType());
    else
      RHS = UnknownVal();
    
    SVal Result = evalBinOp(state, Op, V2, RHS, U->getType());
    
    // Conjure a new symbol if necessary to recover precision.
    if (Result.isUnknown()){
      DefinedOrUnknownSVal SymVal =
        svalBuilder.conjureSymbolVal(0, Ex, LCtx, currBldrCtx->blockCount());
      Result = SymVal;
      
      // If the value is a location, ++/-- should always preserve
      // non-nullness.  Check if the original value was non-null, and if so
      // propagate that constraint.
      if (Loc::isLocType(U->getType())) {
        DefinedOrUnknownSVal Constraint =
        svalBuilder.evalEQ(state, V2,svalBuilder.makeZeroVal(U->getType()));
        
        if (!state->assume(Constraint, true)) {
          // It isn't feasible for the original value to be null.
          // Propagate this constraint.
          Constraint = svalBuilder.evalEQ(state, SymVal,
                                       svalBuilder.makeZeroVal(U->getType()));
          
          
          state = state->assume(Constraint, false);
          assert(state);
        }
      }
    }
    
    // Since the lvalue-to-rvalue conversion is explicit in the AST,
    // we bind an l-value if the operator is prefix and an lvalue (in C++).
    if (U->isGLValue())
      state = state->BindExpr(U, LCtx, loc);
    else
      state = state->BindExpr(U, LCtx, U->isPostfix() ? V2 : Result);
    
    // Perform the store.
    Bldr.takeNodes(*I);
    ExplodedNodeSet Dst3;
    evalStore(Dst3, U, U, *I, state, loc, Result);
    Bldr.addNodes(Dst3);
  }
  Dst.insert(Dst2);
}
