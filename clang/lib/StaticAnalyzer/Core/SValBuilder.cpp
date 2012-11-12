// SValBuilder.cpp - Basic class for all SValBuilder implementations -*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SValBuilder, the base class for all (complete) SValBuilder
//  implementations.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExprCXX.h"
#include "clang/AST/DeclCXX.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/BasicValueFactory.h"

using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// Basic SVal creation.
//===----------------------------------------------------------------------===//

void SValBuilder::anchor() { }

DefinedOrUnknownSVal SValBuilder::makeZeroVal(QualType type) {
  if (Loc::isLocType(type))
    return makeNull();

  if (type->isIntegerType())
    return makeIntVal(0, type);

  // FIXME: Handle floats.
  // FIXME: Handle structs.
  return UnknownVal();
}

NonLoc SValBuilder::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                                const llvm::APSInt& rhs, QualType type) {
  // The Environment ensures we always get a persistent APSInt in
  // BasicValueFactory, so we don't need to get the APSInt from
  // BasicValueFactory again.
  assert(lhs);
  assert(!Loc::isLocType(type));
  return nonloc::SymbolVal(SymMgr.getSymIntExpr(lhs, op, rhs, type));
}

NonLoc SValBuilder::makeNonLoc(const llvm::APSInt& lhs,
                               BinaryOperator::Opcode op, const SymExpr *rhs,
                               QualType type) {
  assert(rhs);
  assert(!Loc::isLocType(type));
  return nonloc::SymbolVal(SymMgr.getIntSymExpr(lhs, op, rhs, type));
}

NonLoc SValBuilder::makeNonLoc(const SymExpr *lhs, BinaryOperator::Opcode op,
                               const SymExpr *rhs, QualType type) {
  assert(lhs && rhs);
  assert(!Loc::isLocType(type));
  return nonloc::SymbolVal(SymMgr.getSymSymExpr(lhs, op, rhs, type));
}

NonLoc SValBuilder::makeNonLoc(const SymExpr *operand,
                               QualType fromTy, QualType toTy) {
  assert(operand);
  assert(!Loc::isLocType(toTy));
  return nonloc::SymbolVal(SymMgr.getCastSymbol(operand, fromTy, toTy));
}

SVal SValBuilder::convertToArrayIndex(SVal val) {
  if (val.isUnknownOrUndef())
    return val;

  // Common case: we have an appropriately sized integer.
  if (nonloc::ConcreteInt* CI = dyn_cast<nonloc::ConcreteInt>(&val)) {
    const llvm::APSInt& I = CI->getValue();
    if (I.getBitWidth() == ArrayIndexWidth && I.isSigned())
      return val;
  }

  return evalCastFromNonLoc(cast<NonLoc>(val), ArrayIndexTy);
}

nonloc::ConcreteInt SValBuilder::makeBoolVal(const CXXBoolLiteralExpr *boolean){
  return makeTruthVal(boolean->getValue());
}

DefinedOrUnknownSVal 
SValBuilder::getRegionValueSymbolVal(const TypedValueRegion* region) {
  QualType T = region->getValueType();

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getRegionValueSymbol(region);

  if (Loc::isLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal SValBuilder::conjureSymbolVal(const void *symbolTag,
                                                   const Expr *expr,
                                                   const LocationContext *LCtx,
                                                   unsigned count) {
  QualType T = expr->getType();
  return conjureSymbolVal(symbolTag, expr, LCtx, T, count);
}

DefinedOrUnknownSVal SValBuilder::conjureSymbolVal(const void *symbolTag,
                                                   const Expr *expr,
                                                   const LocationContext *LCtx,
                                                   QualType type,
                                                   unsigned count) {
  if (!SymbolManager::canSymbolicate(type))
    return UnknownVal();

  SymbolRef sym = SymMgr.conjureSymbol(expr, LCtx, type, count, symbolTag);

  if (Loc::isLocType(type))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}


DefinedOrUnknownSVal SValBuilder::conjureSymbolVal(const Stmt *stmt,
                                                   const LocationContext *LCtx,
                                                   QualType type,
                                                   unsigned visitCount) {
  if (!SymbolManager::canSymbolicate(type))
    return UnknownVal();

  SymbolRef sym = SymMgr.conjureSymbol(stmt, LCtx, type, visitCount);
  
  if (Loc::isLocType(type))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));
  
  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal
SValBuilder::getConjuredHeapSymbolVal(const Expr *E,
                                      const LocationContext *LCtx,
                                      unsigned VisitCount) {
  QualType T = E->getType();
  assert(Loc::isLocType(T));
  assert(SymbolManager::canSymbolicate(T));

  SymbolRef sym = SymMgr.conjureSymbol(E, LCtx, T, VisitCount);
  return loc::MemRegionVal(MemMgr.getSymbolicHeapRegion(sym));
}

DefinedSVal SValBuilder::getMetadataSymbolVal(const void *symbolTag,
                                              const MemRegion *region,
                                              const Expr *expr, QualType type,
                                              unsigned count) {
  assert(SymbolManager::canSymbolicate(type) && "Invalid metadata symbol type");

  SymbolRef sym =
      SymMgr.getMetadataSymbol(region, expr, type, count, symbolTag);

  if (Loc::isLocType(type))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedOrUnknownSVal
SValBuilder::getDerivedRegionValueSymbolVal(SymbolRef parentSymbol,
                                             const TypedValueRegion *region) {
  QualType T = region->getValueType();

  if (!SymbolManager::canSymbolicate(T))
    return UnknownVal();

  SymbolRef sym = SymMgr.getDerivedSymbol(parentSymbol, region);

  if (Loc::isLocType(T))
    return loc::MemRegionVal(MemMgr.getSymbolicRegion(sym));

  return nonloc::SymbolVal(sym);
}

DefinedSVal SValBuilder::getFunctionPointer(const FunctionDecl *func) {
  return loc::MemRegionVal(MemMgr.getFunctionTextRegion(func));
}

DefinedSVal SValBuilder::getBlockPointer(const BlockDecl *block,
                                         CanQualType locTy,
                                         const LocationContext *locContext) {
  const BlockTextRegion *BC =
    MemMgr.getBlockTextRegion(block, locTy, locContext->getAnalysisDeclContext());
  const BlockDataRegion *BD = MemMgr.getBlockDataRegion(BC, locContext);
  return loc::MemRegionVal(BD);
}

/// Return a memory region for the 'this' object reference.
loc::MemRegionVal SValBuilder::getCXXThis(const CXXMethodDecl *D,
                                          const StackFrameContext *SFC) {
  return loc::MemRegionVal(getRegionManager().
                           getCXXThisRegion(D->getThisType(getContext()), SFC));
}

/// Return a memory region for the 'this' object reference.
loc::MemRegionVal SValBuilder::getCXXThis(const CXXRecordDecl *D,
                                          const StackFrameContext *SFC) {
  const Type *T = D->getTypeForDecl();
  QualType PT = getContext().getPointerType(QualType(T, 0));
  return loc::MemRegionVal(getRegionManager().getCXXThisRegion(PT, SFC));
}

//===----------------------------------------------------------------------===//

SVal SValBuilder::makeSymExprValNN(ProgramStateRef State,
                                   BinaryOperator::Opcode Op,
                                   NonLoc LHS, NonLoc RHS,
                                   QualType ResultTy) {
  if (!State->isTainted(RHS) && !State->isTainted(LHS))
    return UnknownVal();
    
  const SymExpr *symLHS = LHS.getAsSymExpr();
  const SymExpr *symRHS = RHS.getAsSymExpr();
  // TODO: When the Max Complexity is reached, we should conjure a symbol
  // instead of generating an Unknown value and propagate the taint info to it.
  const unsigned MaxComp = 10000; // 100000 28X

  if (symLHS && symRHS &&
      (symLHS->computeComplexity() + symRHS->computeComplexity()) <  MaxComp)
    return makeNonLoc(symLHS, Op, symRHS, ResultTy);

  if (symLHS && symLHS->computeComplexity() < MaxComp)
    if (const nonloc::ConcreteInt *rInt = dyn_cast<nonloc::ConcreteInt>(&RHS))
      return makeNonLoc(symLHS, Op, rInt->getValue(), ResultTy);

  if (symRHS && symRHS->computeComplexity() < MaxComp)
    if (const nonloc::ConcreteInt *lInt = dyn_cast<nonloc::ConcreteInt>(&LHS))
      return makeNonLoc(lInt->getValue(), Op, symRHS, ResultTy);

  return UnknownVal();
}


SVal SValBuilder::evalBinOp(ProgramStateRef state, BinaryOperator::Opcode op,
                            SVal lhs, SVal rhs, QualType type) {

  if (lhs.isUndef() || rhs.isUndef())
    return UndefinedVal();

  if (lhs.isUnknown() || rhs.isUnknown())
    return UnknownVal();

  if (isa<Loc>(lhs)) {
    if (isa<Loc>(rhs))
      return evalBinOpLL(state, op, cast<Loc>(lhs), cast<Loc>(rhs), type);

    return evalBinOpLN(state, op, cast<Loc>(lhs), cast<NonLoc>(rhs), type);
  }

  if (isa<Loc>(rhs)) {
    // Support pointer arithmetic where the addend is on the left
    // and the pointer on the right.
    assert(op == BO_Add);

    // Commute the operands.
    return evalBinOpLN(state, op, cast<Loc>(rhs), cast<NonLoc>(lhs), type);
  }

  return evalBinOpNN(state, op, cast<NonLoc>(lhs), cast<NonLoc>(rhs), type);
}

DefinedOrUnknownSVal SValBuilder::evalEQ(ProgramStateRef state,
                                         DefinedOrUnknownSVal lhs,
                                         DefinedOrUnknownSVal rhs) {
  return cast<DefinedOrUnknownSVal>(evalBinOp(state, BO_EQ, lhs, rhs,
                                              Context.IntTy));
}

/// Recursively check if the pointer types are equal modulo const, volatile,
/// and restrict qualifiers. Assumes the input types are canonical.
/// TODO: This is based off of code in SemaCast; can we reuse it.
static bool haveSimilarTypes(ASTContext &Context, QualType T1,
                                                  QualType T2) {
  while (Context.UnwrapSimilarPointerTypes(T1, T2)) {
    Qualifiers Quals1, Quals2;
    T1 = Context.getUnqualifiedArrayType(T1, Quals1);
    T2 = Context.getUnqualifiedArrayType(T2, Quals2);

    // Make sure that non cvr-qualifiers the other qualifiers (e.g., address
    // spaces) are identical.
    Quals1.removeCVRQualifiers();
    Quals2.removeCVRQualifiers();
    if (Quals1 != Quals2)
      return false;
  }

  if (T1 != T2)
    return false;

  return true;
}

// FIXME: should rewrite according to the cast kind.
SVal SValBuilder::evalCast(SVal val, QualType castTy, QualType originalTy) {
  castTy = Context.getCanonicalType(castTy);
  originalTy = Context.getCanonicalType(originalTy);
  if (val.isUnknownOrUndef() || castTy == originalTy)
    return val;

  // For const casts, just propagate the value.
  if (!castTy->isVariableArrayType() && !originalTy->isVariableArrayType())
    if (haveSimilarTypes(Context, Context.getPointerType(castTy),
                                  Context.getPointerType(originalTy)))
      return val;
  
  // Check for casts from pointers to integers.
  if (castTy->isIntegerType() && Loc::isLocType(originalTy))
    return evalCastFromLoc(cast<Loc>(val), castTy);

  // Check for casts from integers to pointers.
  if (Loc::isLocType(castTy) && originalTy->isIntegerType()) {
    if (nonloc::LocAsInteger *LV = dyn_cast<nonloc::LocAsInteger>(&val)) {
      if (const MemRegion *R = LV->getLoc().getAsRegion()) {
        StoreManager &storeMgr = StateMgr.getStoreManager();
        R = storeMgr.castRegion(R, castTy);
        return R ? SVal(loc::MemRegionVal(R)) : UnknownVal();
      }
      return LV->getLoc();
    }
    return dispatchCast(val, castTy);
  }

  // Just pass through function and block pointers.
  if (originalTy->isBlockPointerType() || originalTy->isFunctionPointerType()) {
    assert(Loc::isLocType(castTy));
    return val;
  }

  // Check for casts from array type to another type.
  if (originalTy->isArrayType()) {
    // We will always decay to a pointer.
    val = StateMgr.ArrayToPointer(cast<Loc>(val));

    // Are we casting from an array to a pointer?  If so just pass on
    // the decayed value.
    if (castTy->isPointerType() || castTy->isReferenceType())
      return val;

    // Are we casting from an array to an integer?  If so, cast the decayed
    // pointer value to an integer.
    assert(castTy->isIntegerType());

    // FIXME: Keep these here for now in case we decide soon that we
    // need the original decayed type.
    //    QualType elemTy = cast<ArrayType>(originalTy)->getElementType();
    //    QualType pointerTy = C.getPointerType(elemTy);
    return evalCastFromLoc(cast<Loc>(val), castTy);
  }

  // Check for casts from a region to a specific type.
  if (const MemRegion *R = val.getAsRegion()) {
    // Handle other casts of locations to integers.
    if (castTy->isIntegerType())
      return evalCastFromLoc(loc::MemRegionVal(R), castTy);

    // FIXME: We should handle the case where we strip off view layers to get
    //  to a desugared type.
    if (!Loc::isLocType(castTy)) {
      // FIXME: There can be gross cases where one casts the result of a function
      // (that returns a pointer) to some other value that happens to fit
      // within that pointer value.  We currently have no good way to
      // model such operations.  When this happens, the underlying operation
      // is that the caller is reasoning about bits.  Conceptually we are
      // layering a "view" of a location on top of those bits.  Perhaps
      // we need to be more lazy about mutual possible views, even on an
      // SVal?  This may be necessary for bit-level reasoning as well.
      return UnknownVal();
    }

    // We get a symbolic function pointer for a dereference of a function
    // pointer, but it is of function type. Example:

    //  struct FPRec {
    //    void (*my_func)(int * x);
    //  };
    //
    //  int bar(int x);
    //
    //  int f1_a(struct FPRec* foo) {
    //    int x;
    //    (*foo->my_func)(&x);
    //    return bar(x)+1; // no-warning
    //  }

    assert(Loc::isLocType(originalTy) || originalTy->isFunctionType() ||
           originalTy->isBlockPointerType() || castTy->isReferenceType());

    StoreManager &storeMgr = StateMgr.getStoreManager();

    // Delegate to store manager to get the result of casting a region to a
    // different type.  If the MemRegion* returned is NULL, this expression
    // Evaluates to UnknownVal.
    R = storeMgr.castRegion(R, castTy);
    return R ? SVal(loc::MemRegionVal(R)) : UnknownVal();
  }

  return dispatchCast(val, castTy);
}
