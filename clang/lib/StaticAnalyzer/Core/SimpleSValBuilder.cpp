// SimpleSValBuilder.cpp - A basic SValBuilder -----------------------*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SimpleSValBuilder, a basic implementation of SValBuilder.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"

using namespace clang;
using namespace ento;

namespace {
class SimpleSValBuilder : public SValBuilder {
protected:
  virtual SVal dispatchCast(SVal val, QualType castTy);
  virtual SVal evalCastFromNonLoc(NonLoc val, QualType castTy);
  virtual SVal evalCastFromLoc(Loc val, QualType castTy);

public:
  SimpleSValBuilder(llvm::BumpPtrAllocator &alloc, ASTContext &context,
                    ProgramStateManager &stateMgr)
                    : SValBuilder(alloc, context, stateMgr) {}
  virtual ~SimpleSValBuilder() {}

  virtual SVal evalMinus(NonLoc val);
  virtual SVal evalComplement(NonLoc val);
  virtual SVal evalBinOpNN(ProgramStateRef state, BinaryOperator::Opcode op,
                           NonLoc lhs, NonLoc rhs, QualType resultTy);
  virtual SVal evalBinOpLL(ProgramStateRef state, BinaryOperator::Opcode op,
                           Loc lhs, Loc rhs, QualType resultTy);
  virtual SVal evalBinOpLN(ProgramStateRef state, BinaryOperator::Opcode op,
                           Loc lhs, NonLoc rhs, QualType resultTy);

  /// getKnownValue - evaluates a given SVal. If the SVal has only one possible
  ///  (integer) value, that value is returned. Otherwise, returns NULL.
  virtual const llvm::APSInt *getKnownValue(ProgramStateRef state, SVal V);
  
  SVal MakeSymIntVal(const SymExpr *LHS, BinaryOperator::Opcode op,
                     const llvm::APSInt &RHS, QualType resultTy);
};
} // end anonymous namespace

SValBuilder *ento::createSimpleSValBuilder(llvm::BumpPtrAllocator &alloc,
                                           ASTContext &context,
                                           ProgramStateManager &stateMgr) {
  return new SimpleSValBuilder(alloc, context, stateMgr);
}

//===----------------------------------------------------------------------===//
// Transfer function for Casts.
//===----------------------------------------------------------------------===//

SVal SimpleSValBuilder::dispatchCast(SVal Val, QualType CastTy) {
  assert(isa<Loc>(&Val) || isa<NonLoc>(&Val));
  return isa<Loc>(Val) ? evalCastFromLoc(cast<Loc>(Val), CastTy)
                       : evalCastFromNonLoc(cast<NonLoc>(Val), CastTy);
}

SVal SimpleSValBuilder::evalCastFromNonLoc(NonLoc val, QualType castTy) {

  bool isLocType = Loc::isLocType(castTy);

  if (nonloc::LocAsInteger *LI = dyn_cast<nonloc::LocAsInteger>(&val)) {
    if (isLocType)
      return LI->getLoc();

    // FIXME: Correctly support promotions/truncations.
    unsigned castSize = Context.getTypeSize(castTy);
    if (castSize == LI->getNumBits())
      return val;
    return makeLocAsInteger(LI->getLoc(), castSize);
  }

  if (const SymExpr *se = val.getAsSymbolicExpression()) {
    QualType T = Context.getCanonicalType(se->getType());
    // If types are the same or both are integers, ignore the cast.
    // FIXME: Remove this hack when we support symbolic truncation/extension.
    // HACK: If both castTy and T are integers, ignore the cast.  This is
    // not a permanent solution.  Eventually we want to precisely handle
    // extension/truncation of symbolic integers.  This prevents us from losing
    // precision when we assign 'x = y' and 'y' is symbolic and x and y are
    // different integer types.
   if (haveSameType(T, castTy))
      return val;

    if (!isLocType)
      return makeNonLoc(se, T, castTy);
    return UnknownVal();
  }

  // If value is a non integer constant, produce unknown.
  if (!isa<nonloc::ConcreteInt>(val))
    return UnknownVal();

  // Only handle casts from integers to integers - if val is an integer constant
  // being cast to a non integer type, produce unknown.
  if (!isLocType && !castTy->isIntegerType())
    return UnknownVal();

  llvm::APSInt i = cast<nonloc::ConcreteInt>(val).getValue();
  BasicVals.getAPSIntType(castTy).apply(i);

  if (isLocType)
    return makeIntLocVal(i);
  else
    return makeIntVal(i);
}

SVal SimpleSValBuilder::evalCastFromLoc(Loc val, QualType castTy) {

  // Casts from pointers -> pointers, just return the lval.
  //
  // Casts from pointers -> references, just return the lval.  These
  //   can be introduced by the frontend for corner cases, e.g
  //   casting from va_list* to __builtin_va_list&.
  //
  if (Loc::isLocType(castTy) || castTy->isReferenceType())
    return val;

  // FIXME: Handle transparent unions where a value can be "transparently"
  //  lifted into a union type.
  if (castTy->isUnionType())
    return UnknownVal();

  if (castTy->isIntegerType()) {
    unsigned BitWidth = Context.getTypeSize(castTy);

    if (!isa<loc::ConcreteInt>(val))
      return makeLocAsInteger(val, BitWidth);

    llvm::APSInt i = cast<loc::ConcreteInt>(val).getValue();
    BasicVals.getAPSIntType(castTy).apply(i);
    return makeIntVal(i);
  }

  // All other cases: return 'UnknownVal'.  This includes casting pointers
  // to floats, which is probably badness it itself, but this is a good
  // intermediate solution until we do something better.
  return UnknownVal();
}

//===----------------------------------------------------------------------===//
// Transfer function for unary operators.
//===----------------------------------------------------------------------===//

SVal SimpleSValBuilder::evalMinus(NonLoc val) {
  switch (val.getSubKind()) {
  case nonloc::ConcreteIntKind:
    return cast<nonloc::ConcreteInt>(val).evalMinus(*this);
  default:
    return UnknownVal();
  }
}

SVal SimpleSValBuilder::evalComplement(NonLoc X) {
  switch (X.getSubKind()) {
  case nonloc::ConcreteIntKind:
    return cast<nonloc::ConcreteInt>(X).evalComplement(*this);
  default:
    return UnknownVal();
  }
}

//===----------------------------------------------------------------------===//
// Transfer function for binary operators.
//===----------------------------------------------------------------------===//

static BinaryOperator::Opcode NegateComparison(BinaryOperator::Opcode op) {
  switch (op) {
  default:
    llvm_unreachable("Invalid opcode.");
  case BO_LT: return BO_GE;
  case BO_GT: return BO_LE;
  case BO_LE: return BO_GT;
  case BO_GE: return BO_LT;
  case BO_EQ: return BO_NE;
  case BO_NE: return BO_EQ;
  }
}

static BinaryOperator::Opcode ReverseComparison(BinaryOperator::Opcode op) {
  switch (op) {
  default:
    llvm_unreachable("Invalid opcode.");
  case BO_LT: return BO_GT;
  case BO_GT: return BO_LT;
  case BO_LE: return BO_GE;
  case BO_GE: return BO_LE;
  case BO_EQ:
  case BO_NE:
    return op;
  }
}

SVal SimpleSValBuilder::MakeSymIntVal(const SymExpr *LHS,
                                    BinaryOperator::Opcode op,
                                    const llvm::APSInt &RHS,
                                    QualType resultTy) {
  bool isIdempotent = false;

  // Check for a few special cases with known reductions first.
  switch (op) {
  default:
    // We can't reduce this case; just treat it normally.
    break;
  case BO_Mul:
    // a*0 and a*1
    if (RHS == 0)
      return makeIntVal(0, resultTy);
    else if (RHS == 1)
      isIdempotent = true;
    break;
  case BO_Div:
    // a/0 and a/1
    if (RHS == 0)
      // This is also handled elsewhere.
      return UndefinedVal();
    else if (RHS == 1)
      isIdempotent = true;
    break;
  case BO_Rem:
    // a%0 and a%1
    if (RHS == 0)
      // This is also handled elsewhere.
      return UndefinedVal();
    else if (RHS == 1)
      return makeIntVal(0, resultTy);
    break;
  case BO_Add:
  case BO_Sub:
  case BO_Shl:
  case BO_Shr:
  case BO_Xor:
    // a+0, a-0, a<<0, a>>0, a^0
    if (RHS == 0)
      isIdempotent = true;
    break;
  case BO_And:
    // a&0 and a&(~0)
    if (RHS == 0)
      return makeIntVal(0, resultTy);
    else if (RHS.isAllOnesValue())
      isIdempotent = true;
    break;
  case BO_Or:
    // a|0 and a|(~0)
    if (RHS == 0)
      isIdempotent = true;
    else if (RHS.isAllOnesValue()) {
      const llvm::APSInt &Result = BasicVals.Convert(resultTy, RHS);
      return nonloc::ConcreteInt(Result);
    }
    break;
  }

  // Idempotent ops (like a*1) can still change the type of an expression.
  // Wrap the LHS up in a NonLoc again and let evalCastFromNonLoc do the
  // dirty work.
  if (isIdempotent)
      return evalCastFromNonLoc(nonloc::SymbolVal(LHS), resultTy);

  // If we reach this point, the expression cannot be simplified.
  // Make a SymbolVal for the entire expression, after converting the RHS.
  const llvm::APSInt *ConvertedRHS = &RHS;
  if (BinaryOperator::isComparisonOp(op)) {
    // We're looking for a type big enough to compare the symbolic value
    // with the given constant.
    // FIXME: This is an approximation of Sema::UsualArithmeticConversions.
    ASTContext &Ctx = getContext();
    QualType SymbolType = LHS->getType();
    uint64_t ValWidth = RHS.getBitWidth();
    uint64_t TypeWidth = Ctx.getTypeSize(SymbolType);

    if (ValWidth < TypeWidth) {
      // If the value is too small, extend it.
      ConvertedRHS = &BasicVals.Convert(SymbolType, RHS);
    } else if (ValWidth == TypeWidth) {
      // If the value is signed but the symbol is unsigned, do the comparison
      // in unsigned space. [C99 6.3.1.8]
      // (For the opposite case, the value is already unsigned.)
      if (RHS.isSigned() && !SymbolType->isSignedIntegerOrEnumerationType())
        ConvertedRHS = &BasicVals.Convert(SymbolType, RHS);
    }
  } else
    ConvertedRHS = &BasicVals.Convert(resultTy, RHS);

  return makeNonLoc(LHS, op, *ConvertedRHS, resultTy);
}

SVal SimpleSValBuilder::evalBinOpNN(ProgramStateRef state,
                                  BinaryOperator::Opcode op,
                                  NonLoc lhs, NonLoc rhs,
                                  QualType resultTy)  {
  NonLoc InputLHS = lhs;
  NonLoc InputRHS = rhs;

  // Handle trivial case where left-side and right-side are the same.
  if (lhs == rhs)
    switch (op) {
      default:
        break;
      case BO_EQ:
      case BO_LE:
      case BO_GE:
        return makeTruthVal(true, resultTy);
      case BO_LT:
      case BO_GT:
      case BO_NE:
        return makeTruthVal(false, resultTy);
      case BO_Xor:
      case BO_Sub:
        if (resultTy->isIntegralOrEnumerationType())
          return makeIntVal(0, resultTy);
        return evalCastFromNonLoc(makeIntVal(0, /*Unsigned=*/false), resultTy);
      case BO_Or:
      case BO_And:
        return evalCastFromNonLoc(lhs, resultTy);
    }

  while (1) {
    switch (lhs.getSubKind()) {
    default:
      return makeSymExprValNN(state, op, lhs, rhs, resultTy);
    case nonloc::LocAsIntegerKind: {
      Loc lhsL = cast<nonloc::LocAsInteger>(lhs).getLoc();
      switch (rhs.getSubKind()) {
        case nonloc::LocAsIntegerKind:
          return evalBinOpLL(state, op, lhsL,
                             cast<nonloc::LocAsInteger>(rhs).getLoc(),
                             resultTy);
        case nonloc::ConcreteIntKind: {
          // Transform the integer into a location and compare.
          llvm::APSInt i = cast<nonloc::ConcreteInt>(rhs).getValue();
          BasicVals.getAPSIntType(Context.VoidPtrTy).apply(i);
          return evalBinOpLL(state, op, lhsL, makeLoc(i), resultTy);
        }
        default:
          switch (op) {
            case BO_EQ:
              return makeTruthVal(false, resultTy);
            case BO_NE:
              return makeTruthVal(true, resultTy);
            default:
              // This case also handles pointer arithmetic.
              return makeSymExprValNN(state, op, InputLHS, InputRHS, resultTy);
          }
      }
    }
    case nonloc::ConcreteIntKind: {
      llvm::APSInt LHSValue = cast<nonloc::ConcreteInt>(lhs).getValue();

      // If we're dealing with two known constants, just perform the operation.
      if (const llvm::APSInt *KnownRHSValue = getKnownValue(state, rhs)) {
        llvm::APSInt RHSValue = *KnownRHSValue;
        if (BinaryOperator::isComparisonOp(op)) {
          // We're looking for a type big enough to compare the two values.
          // FIXME: This is not correct. char + short will result in a promotion
          // to int. Unfortunately we have lost types by this point.
          APSIntType CompareType = std::max(APSIntType(LHSValue),
                                            APSIntType(RHSValue));
          CompareType.apply(LHSValue);
          CompareType.apply(RHSValue);
        } else if (!BinaryOperator::isShiftOp(op)) {
          APSIntType IntType = BasicVals.getAPSIntType(resultTy);
          IntType.apply(LHSValue);
          IntType.apply(RHSValue);
        }

        const llvm::APSInt *Result =
          BasicVals.evalAPSInt(op, LHSValue, RHSValue);
        if (!Result)
          return UndefinedVal();

        return nonloc::ConcreteInt(*Result);
      }

      // Swap the left and right sides and flip the operator if doing so
      // allows us to better reason about the expression (this is a form
      // of expression canonicalization).
      // While we're at it, catch some special cases for non-commutative ops.
      switch (op) {
      case BO_LT:
      case BO_GT:
      case BO_LE:
      case BO_GE:
        op = ReverseComparison(op);
        // FALL-THROUGH
      case BO_EQ:
      case BO_NE:
      case BO_Add:
      case BO_Mul:
      case BO_And:
      case BO_Xor:
      case BO_Or:
        std::swap(lhs, rhs);
        continue;
      case BO_Shr:
        // (~0)>>a
        if (LHSValue.isAllOnesValue() && LHSValue.isSigned())
          return evalCastFromNonLoc(lhs, resultTy);
        // FALL-THROUGH
      case BO_Shl:
        // 0<<a and 0>>a
        if (LHSValue == 0)
          return evalCastFromNonLoc(lhs, resultTy);
        return makeSymExprValNN(state, op, InputLHS, InputRHS, resultTy);
      default:
        return makeSymExprValNN(state, op, InputLHS, InputRHS, resultTy);
      }
    }
    case nonloc::SymbolValKind: {
      // We only handle LHS as simple symbols or SymIntExprs.
      SymbolRef Sym = cast<nonloc::SymbolVal>(lhs).getSymbol();

      // LHS is a symbolic expression.
      if (const SymIntExpr *symIntExpr = dyn_cast<SymIntExpr>(Sym)) {

        // Is this a logical not? (!x is represented as x == 0.)
        if (op == BO_EQ && rhs.isZeroConstant()) {
          // We know how to negate certain expressions. Simplify them here.

          BinaryOperator::Opcode opc = symIntExpr->getOpcode();
          switch (opc) {
          default:
            // We don't know how to negate this operation.
            // Just handle it as if it were a normal comparison to 0.
            break;
          case BO_LAnd:
          case BO_LOr:
            llvm_unreachable("Logical operators handled by branching logic.");
          case BO_Assign:
          case BO_MulAssign:
          case BO_DivAssign:
          case BO_RemAssign:
          case BO_AddAssign:
          case BO_SubAssign:
          case BO_ShlAssign:
          case BO_ShrAssign:
          case BO_AndAssign:
          case BO_XorAssign:
          case BO_OrAssign:
          case BO_Comma:
            llvm_unreachable("'=' and ',' operators handled by ExprEngine.");
          case BO_PtrMemD:
          case BO_PtrMemI:
            llvm_unreachable("Pointer arithmetic not handled here.");
          case BO_LT:
          case BO_GT:
          case BO_LE:
          case BO_GE:
          case BO_EQ:
          case BO_NE:
            // Negate the comparison and make a value.
            opc = NegateComparison(opc);
            assert(symIntExpr->getType() == resultTy);
            return makeNonLoc(symIntExpr->getLHS(), opc,
                symIntExpr->getRHS(), resultTy);
          }
        }

        // For now, only handle expressions whose RHS is a constant.
        if (const llvm::APSInt *RHSValue = getKnownValue(state, rhs)) {
          // If both the LHS and the current expression are additive,
          // fold their constants and try again.
          if (BinaryOperator::isAdditiveOp(op)) {
            BinaryOperator::Opcode lop = symIntExpr->getOpcode();
            if (BinaryOperator::isAdditiveOp(lop)) {
              // Convert the two constants to a common type, then combine them.

              // resultTy may not be the best type to convert to, but it's
              // probably the best choice in expressions with mixed type
              // (such as x+1U+2LL). The rules for implicit conversions should
              // choose a reasonable type to preserve the expression, and will
              // at least match how the value is going to be used.
              APSIntType IntType = BasicVals.getAPSIntType(resultTy);
              const llvm::APSInt &first = IntType.convert(symIntExpr->getRHS());
              const llvm::APSInt &second = IntType.convert(*RHSValue);

              const llvm::APSInt *newRHS;
              if (lop == op)
                newRHS = BasicVals.evalAPSInt(BO_Add, first, second);
              else
                newRHS = BasicVals.evalAPSInt(BO_Sub, first, second);

              assert(newRHS && "Invalid operation despite common type!");
              rhs = nonloc::ConcreteInt(*newRHS);
              lhs = nonloc::SymbolVal(symIntExpr->getLHS());
              op = lop;
              continue;
            }
          }

          // Otherwise, make a SymIntExpr out of the expression.
          return MakeSymIntVal(symIntExpr, op, *RHSValue, resultTy);
        }


      } else if (isa<SymbolData>(Sym)) {
        // Does the symbol simplify to a constant?  If so, "fold" the constant
        // by setting 'lhs' to a ConcreteInt and try again.
        if (const llvm::APSInt *Constant = state->getConstraintManager()
                                                  .getSymVal(state, Sym)) {
          lhs = nonloc::ConcreteInt(*Constant);
          continue;
        }

        // Is the RHS a constant?
        if (const llvm::APSInt *RHSValue = getKnownValue(state, rhs))
          return MakeSymIntVal(Sym, op, *RHSValue, resultTy);
      }

      // Give up -- this is not a symbolic expression we can handle.
      return makeSymExprValNN(state, op, InputLHS, InputRHS, resultTy);
    }
    }
  }
}

// FIXME: all this logic will change if/when we have MemRegion::getLocation().
SVal SimpleSValBuilder::evalBinOpLL(ProgramStateRef state,
                                  BinaryOperator::Opcode op,
                                  Loc lhs, Loc rhs,
                                  QualType resultTy) {
  // Only comparisons and subtractions are valid operations on two pointers.
  // See [C99 6.5.5 through 6.5.14] or [C++0x 5.6 through 5.15].
  // However, if a pointer is casted to an integer, evalBinOpNN may end up
  // calling this function with another operation (PR7527). We don't attempt to
  // model this for now, but it could be useful, particularly when the
  // "location" is actually an integer value that's been passed through a void*.
  if (!(BinaryOperator::isComparisonOp(op) || op == BO_Sub))
    return UnknownVal();

  // Special cases for when both sides are identical.
  if (lhs == rhs) {
    switch (op) {
    default:
      llvm_unreachable("Unimplemented operation for two identical values");
    case BO_Sub:
      return makeZeroVal(resultTy);
    case BO_EQ:
    case BO_LE:
    case BO_GE:
      return makeTruthVal(true, resultTy);
    case BO_NE:
    case BO_LT:
    case BO_GT:
      return makeTruthVal(false, resultTy);
    }
  }

  switch (lhs.getSubKind()) {
  default:
    llvm_unreachable("Ordering not implemented for this Loc.");

  case loc::GotoLabelKind:
    // The only thing we know about labels is that they're non-null.
    if (rhs.isZeroConstant()) {
      switch (op) {
      default:
        break;
      case BO_Sub:
        return evalCastFromLoc(lhs, resultTy);
      case BO_EQ:
      case BO_LE:
      case BO_LT:
        return makeTruthVal(false, resultTy);
      case BO_NE:
      case BO_GT:
      case BO_GE:
        return makeTruthVal(true, resultTy);
      }
    }
    // There may be two labels for the same location, and a function region may
    // have the same address as a label at the start of the function (depending
    // on the ABI).
    // FIXME: we can probably do a comparison against other MemRegions, though.
    // FIXME: is there a way to tell if two labels refer to the same location?
    return UnknownVal(); 

  case loc::ConcreteIntKind: {
    // If one of the operands is a symbol and the other is a constant,
    // build an expression for use by the constraint manager.
    if (SymbolRef rSym = rhs.getAsLocSymbol()) {
      // We can only build expressions with symbols on the left,
      // so we need a reversible operator.
      if (!BinaryOperator::isComparisonOp(op))
        return UnknownVal();

      const llvm::APSInt &lVal = cast<loc::ConcreteInt>(lhs).getValue();
      return makeNonLoc(rSym, ReverseComparison(op), lVal, resultTy);
    }

    // If both operands are constants, just perform the operation.
    if (loc::ConcreteInt *rInt = dyn_cast<loc::ConcreteInt>(&rhs)) {
      SVal ResultVal = cast<loc::ConcreteInt>(lhs).evalBinOp(BasicVals, op,
                                                             *rInt);
      if (Loc *Result = dyn_cast<Loc>(&ResultVal))
        return evalCastFromLoc(*Result, resultTy);
      else
        return UnknownVal();
    }

    // Special case comparisons against NULL.
    // This must come after the test if the RHS is a symbol, which is used to
    // build constraints. The address of any non-symbolic region is guaranteed
    // to be non-NULL, as is any label.
    assert(isa<loc::MemRegionVal>(rhs) || isa<loc::GotoLabel>(rhs));
    if (lhs.isZeroConstant()) {
      switch (op) {
      default:
        break;
      case BO_EQ:
      case BO_GT:
      case BO_GE:
        return makeTruthVal(false, resultTy);
      case BO_NE:
      case BO_LT:
      case BO_LE:
        return makeTruthVal(true, resultTy);
      }
    }

    // Comparing an arbitrary integer to a region or label address is
    // completely unknowable.
    return UnknownVal();
  }
  case loc::MemRegionKind: {
    if (loc::ConcreteInt *rInt = dyn_cast<loc::ConcreteInt>(&rhs)) {
      // If one of the operands is a symbol and the other is a constant,
      // build an expression for use by the constraint manager.
      if (SymbolRef lSym = lhs.getAsLocSymbol())
        return MakeSymIntVal(lSym, op, rInt->getValue(), resultTy);

      // Special case comparisons to NULL.
      // This must come after the test if the LHS is a symbol, which is used to
      // build constraints. The address of any non-symbolic region is guaranteed
      // to be non-NULL.
      if (rInt->isZeroConstant()) {
        switch (op) {
        default:
          break;
        case BO_Sub:
          return evalCastFromLoc(lhs, resultTy);
        case BO_EQ:
        case BO_LT:
        case BO_LE:
          return makeTruthVal(false, resultTy);
        case BO_NE:
        case BO_GT:
        case BO_GE:
          return makeTruthVal(true, resultTy);
        }
      }

      // Comparing a region to an arbitrary integer is completely unknowable.
      return UnknownVal();
    }

    // Get both values as regions, if possible.
    const MemRegion *LeftMR = lhs.getAsRegion();
    assert(LeftMR && "MemRegionKind SVal doesn't have a region!");

    const MemRegion *RightMR = rhs.getAsRegion();
    if (!RightMR)
      // The RHS is probably a label, which in theory could address a region.
      // FIXME: we can probably make a more useful statement about non-code
      // regions, though.
      return UnknownVal();

    const MemSpaceRegion *LeftMS = LeftMR->getMemorySpace();
    const MemSpaceRegion *RightMS = RightMR->getMemorySpace();
    const MemSpaceRegion *UnknownMS = MemMgr.getUnknownRegion();
    const MemRegion *LeftBase = LeftMR->getBaseRegion();
    const MemRegion *RightBase = RightMR->getBaseRegion();

    // If the two regions are from different known memory spaces they cannot be
    // equal. Also, assume that no symbolic region (whose memory space is
    // unknown) is on the stack.
    if (LeftMS != RightMS &&
        ((LeftMS != UnknownMS && RightMS != UnknownMS) ||
         (isa<StackSpaceRegion>(LeftMS) || isa<StackSpaceRegion>(RightMS)))) {
      switch (op) {
      default:
        return UnknownVal();
      case BO_EQ:
        return makeTruthVal(false, resultTy);
      case BO_NE:
        return makeTruthVal(true, resultTy);
      }
    }

    // If both values wrap regions, see if they're from different base regions.
    // Note, heap base symbolic regions are assumed to not alias with
    // each other; for example, we assume that malloc returns different address
    // on each invocation.
    if (LeftBase != RightBase &&
        ((!isa<SymbolicRegion>(LeftBase) && !isa<SymbolicRegion>(RightBase)) ||
         (isa<HeapSpaceRegion>(LeftMS) || isa<HeapSpaceRegion>(RightMS))) ){
      switch (op) {
      default:
        return UnknownVal();
      case BO_EQ:
        return makeTruthVal(false, resultTy);
      case BO_NE:
        return makeTruthVal(true, resultTy);
      }
    }

    // FIXME: If/when there is a getAsRawOffset() for FieldRegions, this
    // ElementRegion path and the FieldRegion path below should be unified.
    if (const ElementRegion *LeftER = dyn_cast<ElementRegion>(LeftMR)) {
      // First see if the right region is also an ElementRegion.
      const ElementRegion *RightER = dyn_cast<ElementRegion>(RightMR);
      if (!RightER)
        return UnknownVal();

      // Next, see if the two ERs have the same super-region and matching types.
      // FIXME: This should do something useful even if the types don't match,
      // though if both indexes are constant the RegionRawOffset path will
      // give the correct answer.
      if (LeftER->getSuperRegion() == RightER->getSuperRegion() &&
          LeftER->getElementType() == RightER->getElementType()) {
        // Get the left index and cast it to the correct type.
        // If the index is unknown or undefined, bail out here.
        SVal LeftIndexVal = LeftER->getIndex();
        NonLoc *LeftIndex = dyn_cast<NonLoc>(&LeftIndexVal);
        if (!LeftIndex)
          return UnknownVal();
        LeftIndexVal = evalCastFromNonLoc(*LeftIndex, resultTy);
        LeftIndex = dyn_cast<NonLoc>(&LeftIndexVal);
        if (!LeftIndex)
          return UnknownVal();

        // Do the same for the right index.
        SVal RightIndexVal = RightER->getIndex();
        NonLoc *RightIndex = dyn_cast<NonLoc>(&RightIndexVal);
        if (!RightIndex)
          return UnknownVal();
        RightIndexVal = evalCastFromNonLoc(*RightIndex, resultTy);
        RightIndex = dyn_cast<NonLoc>(&RightIndexVal);
        if (!RightIndex)
          return UnknownVal();

        // Actually perform the operation.
        // evalBinOpNN expects the two indexes to already be the right type.
        return evalBinOpNN(state, op, *LeftIndex, *RightIndex, resultTy);
      }

      // If the element indexes aren't comparable, see if the raw offsets are.
      RegionRawOffset LeftOffset = LeftER->getAsArrayOffset();
      RegionRawOffset RightOffset = RightER->getAsArrayOffset();

      if (LeftOffset.getRegion() != NULL &&
          LeftOffset.getRegion() == RightOffset.getRegion()) {
        CharUnits left = LeftOffset.getOffset();
        CharUnits right = RightOffset.getOffset();

        switch (op) {
        default:
          return UnknownVal();
        case BO_LT:
          return makeTruthVal(left < right, resultTy);
        case BO_GT:
          return makeTruthVal(left > right, resultTy);
        case BO_LE:
          return makeTruthVal(left <= right, resultTy);
        case BO_GE:
          return makeTruthVal(left >= right, resultTy);
        case BO_EQ:
          return makeTruthVal(left == right, resultTy);
        case BO_NE:
          return makeTruthVal(left != right, resultTy);
        }
      }

      // If we get here, we have no way of comparing the ElementRegions.
      return UnknownVal();
    }

    // See if both regions are fields of the same structure.
    // FIXME: This doesn't handle nesting, inheritance, or Objective-C ivars.
    if (const FieldRegion *LeftFR = dyn_cast<FieldRegion>(LeftMR)) {
      // Only comparisons are meaningful here!
      if (!BinaryOperator::isComparisonOp(op))
        return UnknownVal();

      // First see if the right region is also a FieldRegion.
      const FieldRegion *RightFR = dyn_cast<FieldRegion>(RightMR);
      if (!RightFR)
        return UnknownVal();

      // Next, see if the two FRs have the same super-region.
      // FIXME: This doesn't handle casts yet, and simply stripping the casts
      // doesn't help.
      if (LeftFR->getSuperRegion() != RightFR->getSuperRegion())
        return UnknownVal();

      const FieldDecl *LeftFD = LeftFR->getDecl();
      const FieldDecl *RightFD = RightFR->getDecl();
      const RecordDecl *RD = LeftFD->getParent();

      // Make sure the two FRs are from the same kind of record. Just in case!
      // FIXME: This is probably where inheritance would be a problem.
      if (RD != RightFD->getParent())
        return UnknownVal();

      // We know for sure that the two fields are not the same, since that
      // would have given us the same SVal.
      if (op == BO_EQ)
        return makeTruthVal(false, resultTy);
      if (op == BO_NE)
        return makeTruthVal(true, resultTy);

      // Iterate through the fields and see which one comes first.
      // [C99 6.7.2.1.13] "Within a structure object, the non-bit-field
      // members and the units in which bit-fields reside have addresses that
      // increase in the order in which they are declared."
      bool leftFirst = (op == BO_LT || op == BO_LE);
      for (RecordDecl::field_iterator I = RD->field_begin(),
           E = RD->field_end(); I!=E; ++I) {
        if (*I == LeftFD)
          return makeTruthVal(leftFirst, resultTy);
        if (*I == RightFD)
          return makeTruthVal(!leftFirst, resultTy);
      }

      llvm_unreachable("Fields not found in parent record's definition");
    }

    // If we get here, we have no way of comparing the regions.
    return UnknownVal();
  }
  }
}

SVal SimpleSValBuilder::evalBinOpLN(ProgramStateRef state,
                                  BinaryOperator::Opcode op,
                                  Loc lhs, NonLoc rhs, QualType resultTy) {
  
  // Special case: rhs is a zero constant.
  if (rhs.isZeroConstant())
    return lhs;
  
  // Special case: 'rhs' is an integer that has the same width as a pointer and
  // we are using the integer location in a comparison.  Normally this cannot be
  // triggered, but transfer functions like those for OSCommpareAndSwapBarrier32
  // can generate comparisons that trigger this code.
  // FIXME: Are all locations guaranteed to have pointer width?
  if (BinaryOperator::isComparisonOp(op)) {
    if (nonloc::ConcreteInt *rhsInt = dyn_cast<nonloc::ConcreteInt>(&rhs)) {
      const llvm::APSInt *x = &rhsInt->getValue();
      ASTContext &ctx = Context;
      if (ctx.getTypeSize(ctx.VoidPtrTy) == x->getBitWidth()) {
        // Convert the signedness of the integer (if necessary).
        if (x->isSigned())
          x = &getBasicValueFactory().getValue(*x, true);

        return evalBinOpLL(state, op, lhs, loc::ConcreteInt(*x), resultTy);
      }
    }
    return UnknownVal();
  }
  
  // We are dealing with pointer arithmetic.

  // Handle pointer arithmetic on constant values.
  if (nonloc::ConcreteInt *rhsInt = dyn_cast<nonloc::ConcreteInt>(&rhs)) {
    if (loc::ConcreteInt *lhsInt = dyn_cast<loc::ConcreteInt>(&lhs)) {
      const llvm::APSInt &leftI = lhsInt->getValue();
      assert(leftI.isUnsigned());
      llvm::APSInt rightI(rhsInt->getValue(), /* isUnsigned */ true);

      // Convert the bitwidth of rightI.  This should deal with overflow
      // since we are dealing with concrete values.
      rightI = rightI.extOrTrunc(leftI.getBitWidth());

      // Offset the increment by the pointer size.
      llvm::APSInt Multiplicand(rightI.getBitWidth(), /* isUnsigned */ true);
      rightI *= Multiplicand;
      
      // Compute the adjusted pointer.
      switch (op) {
        case BO_Add:
          rightI = leftI + rightI;
          break;
        case BO_Sub:
          rightI = leftI - rightI;
          break;
        default:
          llvm_unreachable("Invalid pointer arithmetic operation");
      }
      return loc::ConcreteInt(getBasicValueFactory().getValue(rightI));
    }
  }

  // Handle cases where 'lhs' is a region.
  if (const MemRegion *region = lhs.getAsRegion()) {
    rhs = cast<NonLoc>(convertToArrayIndex(rhs));
    SVal index = UnknownVal();
    const MemRegion *superR = 0;
    QualType elementType;

    if (const ElementRegion *elemReg = dyn_cast<ElementRegion>(region)) {
      assert(op == BO_Add || op == BO_Sub);
      index = evalBinOpNN(state, op, elemReg->getIndex(), rhs,
                          getArrayIndexType());
      superR = elemReg->getSuperRegion();
      elementType = elemReg->getElementType();
    }
    else if (isa<SubRegion>(region)) {
      superR = region;
      index = rhs;
      if (resultTy->isAnyPointerType())
        elementType = resultTy->getPointeeType();
    }

    if (NonLoc *indexV = dyn_cast<NonLoc>(&index)) {
      return loc::MemRegionVal(MemMgr.getElementRegion(elementType, *indexV,
                                                       superR, getContext()));
    }
  }
  return UnknownVal();  
}

const llvm::APSInt *SimpleSValBuilder::getKnownValue(ProgramStateRef state,
                                                   SVal V) {
  if (V.isUnknownOrUndef())
    return NULL;

  if (loc::ConcreteInt* X = dyn_cast<loc::ConcreteInt>(&V))
    return &X->getValue();

  if (nonloc::ConcreteInt* X = dyn_cast<nonloc::ConcreteInt>(&V))
    return &X->getValue();

  if (SymbolRef Sym = V.getAsSymbol())
    return state->getConstraintManager().getSymVal(state, Sym);

  // FIXME: Add support for SymExprs.
  return NULL;
}
