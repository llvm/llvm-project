//===--- CompilerComplex.cpp.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ByteCodeEmitter.h"
#include "Compiler.h"
#include "Context.h"
#include "Floating.h"
#include "Function.h"
#include "InterpShared.h"
#include "PrimType.h"
#include "Program.h"
#include "clang/AST/Attr.h"

using namespace clang;
using namespace clang::interp;

template <class Emitter>
bool Compiler<Emitter>::VisitComplexBinOp(const BinaryOperator *E) {
  // Prepare storage for result.
  if (!Initializing) {
    std::optional<unsigned> LocalIndex = allocateLocal(E);
    if (!LocalIndex)
      return false;
    if (!this->emitGetPtrLocal(*LocalIndex, E))
      return false;
  }

  // Both LHS and RHS might _not_ be of complex type, but one of them
  // needs to be.
  const Expr *LHS = E->getLHS();
  const Expr *RHS = E->getRHS();

  PrimType ResultElemT = this->classifyComplexElementType(E->getType());
  unsigned ResultOffset = ~0u;
  if (!DiscardResult)
    ResultOffset = this->allocateLocalPrimitive(E, PT_Ptr, true, false);

  // Save result pointer in ResultOffset
  if (!this->DiscardResult) {
    if (!this->emitDupPtr(E))
      return false;
    if (!this->emitSetLocal(PT_Ptr, ResultOffset, E))
      return false;
  }
  QualType LHSType = LHS->getType();
  if (const auto *AT = LHSType->getAs<AtomicType>())
    LHSType = AT->getValueType();
  QualType RHSType = RHS->getType();
  if (const auto *AT = RHSType->getAs<AtomicType>())
    RHSType = AT->getValueType();

  bool LHSIsComplex = LHSType->isAnyComplexType();
  unsigned LHSOffset;
  bool RHSIsComplex = RHSType->isAnyComplexType();

  // For ComplexComplex Mul, we have special ops to make their implementation
  // easier.
  BinaryOperatorKind Op = E->getOpcode();
  if (Op == BO_Mul && LHSIsComplex && RHSIsComplex) {
    assert(classifyPrim(LHSType->getAs<ComplexType>()->getElementType()) ==
           classifyPrim(RHSType->getAs<ComplexType>()->getElementType()));
    PrimType ElemT =
        classifyPrim(LHSType->getAs<ComplexType>()->getElementType());
    if (!this->visit(LHS))
      return false;
    if (!this->visit(RHS))
      return false;
    return this->emitMulc(ElemT, E);
  }

  if (Op == BO_Div && RHSIsComplex) {
    QualType ElemQT = RHSType->getAs<ComplexType>()->getElementType();
    PrimType ElemT = classifyPrim(ElemQT);
    // If the LHS is not complex, we still need to do the full complex
    // division, so just stub create a complex value and stub it out with
    // the LHS and a zero.

    if (!LHSIsComplex) {
      // This is using the RHS type for the fake-complex LHS.
      if (auto LHSO = allocateLocal(RHS))
        LHSOffset = *LHSO;
      else
        return false;

      if (!this->emitGetPtrLocal(LHSOffset, E))
        return false;

      if (!this->visit(LHS))
        return false;
      // real is LHS
      if (!this->emitInitElem(ElemT, 0, E))
        return false;
      // imag is zero
      if (!this->visitZeroInitializer(ElemT, ElemQT, E))
        return false;
      if (!this->emitInitElem(ElemT, 1, E))
        return false;
    } else {
      if (!this->visit(LHS))
        return false;
    }

    if (!this->visit(RHS))
      return false;
    return this->emitDivc(ElemT, E);
  }

  // Evaluate LHS and save value to LHSOffset.
  if (LHSType->isAnyComplexType()) {
    LHSOffset = this->allocateLocalPrimitive(LHS, PT_Ptr, true, false);
    if (!this->visit(LHS))
      return false;
    if (!this->emitSetLocal(PT_Ptr, LHSOffset, E))
      return false;
  } else {
    PrimType LHST = classifyPrim(LHSType);
    LHSOffset = this->allocateLocalPrimitive(LHS, LHST, true, false);
    if (!this->visit(LHS))
      return false;
    if (!this->emitSetLocal(LHST, LHSOffset, E))
      return false;
  }

  // Same with RHS.
  unsigned RHSOffset;
  if (RHSType->isAnyComplexType()) {
    RHSOffset = this->allocateLocalPrimitive(RHS, PT_Ptr, true, false);
    if (!this->visit(RHS))
      return false;
    if (!this->emitSetLocal(PT_Ptr, RHSOffset, E))
      return false;
  } else {
    PrimType RHST = classifyPrim(RHSType);
    RHSOffset = this->allocateLocalPrimitive(RHS, RHST, true, false);
    if (!this->visit(RHS))
      return false;
    if (!this->emitSetLocal(RHST, RHSOffset, E))
      return false;
  }

  // For both LHS and RHS, either load the value from the complex pointer, or
  // directly from the local variable. For index 1 (i.e. the imaginary part),
  // just load 0 and do the operation anyway.
  auto loadComplexValue = [this](bool IsComplex, bool LoadZero,
                                 unsigned ElemIndex, unsigned Offset,
                                 const Expr *E) -> bool {
    if (IsComplex) {
      if (!this->emitGetLocal(PT_Ptr, Offset, E))
        return false;
      return this->emitArrayElemPop(classifyComplexElementType(E->getType()),
                                    ElemIndex, E);
    }
    if (ElemIndex == 0 || !LoadZero)
      return this->emitGetLocal(classifyPrim(E->getType()), Offset, E);
    return this->visitZeroInitializer(classifyPrim(E->getType()), E->getType(),
                                      E);
  };

  // Now we can get pointers to the LHS and RHS from the offsets above.
  for (unsigned ElemIndex = 0; ElemIndex != 2; ++ElemIndex) {
    // Result pointer for the store later.
    if (!this->DiscardResult) {
      if (!this->emitGetLocal(PT_Ptr, ResultOffset, E))
        return false;
    }

    // The actual operation.
    switch (Op) {
    case BO_Add:
      if (!loadComplexValue(LHSIsComplex, true, ElemIndex, LHSOffset, LHS))
        return false;

      if (!loadComplexValue(RHSIsComplex, true, ElemIndex, RHSOffset, RHS))
        return false;
      if (ResultElemT == PT_Float) {
        if (!this->emitAddf(getRoundingMode(E), E))
          return false;
      } else {
        if (!this->emitAdd(ResultElemT, E))
          return false;
      }
      break;
    case BO_Sub:
      if (!loadComplexValue(LHSIsComplex, true, ElemIndex, LHSOffset, LHS))
        return false;

      if (!loadComplexValue(RHSIsComplex, true, ElemIndex, RHSOffset, RHS))
        return false;
      if (ResultElemT == PT_Float) {
        if (!this->emitSubf(getRoundingMode(E), E))
          return false;
      } else {
        if (!this->emitSub(ResultElemT, E))
          return false;
      }
      break;
    case BO_Mul:
      if (!loadComplexValue(LHSIsComplex, false, ElemIndex, LHSOffset, LHS))
        return false;

      if (!loadComplexValue(RHSIsComplex, false, ElemIndex, RHSOffset, RHS))
        return false;

      if (ResultElemT == PT_Float) {
        if (!this->emitMulf(getRoundingMode(E), E))
          return false;
      } else {
        if (!this->emitMul(ResultElemT, E))
          return false;
      }
      break;
    case BO_Div:
      assert(!RHSIsComplex);
      if (!loadComplexValue(LHSIsComplex, false, ElemIndex, LHSOffset, LHS))
        return false;

      if (!loadComplexValue(RHSIsComplex, false, ElemIndex, RHSOffset, RHS))
        return false;

      if (ResultElemT == PT_Float) {
        if (!this->emitDivf(getRoundingMode(E), E))
          return false;
      } else {
        if (!this->emitDiv(ResultElemT, E))
          return false;
      }
      break;

    default:
      return false;
    }

    if (!this->DiscardResult) {
      // Initialize array element with the value we just computed.
      if (!this->emitInitElemPop(ResultElemT, ElemIndex, E))
        return false;
    } else {
      if (!this->emitPop(ResultElemT, E))
        return false;
    }
  }
  return true;
}

template <class Emitter>
bool Compiler<Emitter>::emitComplexComparison(const Expr *LHS, const Expr *RHS,
                                              const BinaryOperator *E) {
  assert(E->isComparisonOp());
  assert(!Initializing);
  assert(!DiscardResult);

  PrimType ElemT;
  bool LHSIsComplex;
  unsigned LHSOffset;
  if (LHS->getType()->isAnyComplexType()) {
    LHSIsComplex = true;
    ElemT = classifyComplexElementType(LHS->getType());
    LHSOffset = allocateLocalPrimitive(LHS, PT_Ptr, /*IsConst=*/true,
                                       /*IsExtended=*/false);
    if (!this->visit(LHS))
      return false;
    if (!this->emitSetLocal(PT_Ptr, LHSOffset, E))
      return false;
  } else {
    LHSIsComplex = false;
    PrimType LHST = classifyPrim(LHS->getType());
    LHSOffset = this->allocateLocalPrimitive(LHS, LHST, true, false);
    if (!this->visit(LHS))
      return false;
    if (!this->emitSetLocal(LHST, LHSOffset, E))
      return false;
  }

  bool RHSIsComplex;
  unsigned RHSOffset;
  if (RHS->getType()->isAnyComplexType()) {
    RHSIsComplex = true;
    ElemT = classifyComplexElementType(RHS->getType());
    RHSOffset = allocateLocalPrimitive(RHS, PT_Ptr, /*IsConst=*/true,
                                       /*IsExtended=*/false);
    if (!this->visit(RHS))
      return false;
    if (!this->emitSetLocal(PT_Ptr, RHSOffset, E))
      return false;
  } else {
    RHSIsComplex = false;
    PrimType RHST = classifyPrim(RHS->getType());
    RHSOffset = this->allocateLocalPrimitive(RHS, RHST, true, false);
    if (!this->visit(RHS))
      return false;
    if (!this->emitSetLocal(RHST, RHSOffset, E))
      return false;
  }

  auto getElem = [&](unsigned LocalOffset, unsigned Index,
                     bool IsComplex) -> bool {
    if (IsComplex) {
      if (!this->emitGetLocal(PT_Ptr, LocalOffset, E))
        return false;
      return this->emitArrayElemPop(ElemT, Index, E);
    }
    return this->emitGetLocal(ElemT, LocalOffset, E);
  };

  for (unsigned I = 0; I != 2; ++I) {
    // Get both values.
    if (!getElem(LHSOffset, I, LHSIsComplex))
      return false;
    if (!getElem(RHSOffset, I, RHSIsComplex))
      return false;
    // And compare them.
    if (!this->emitEQ(ElemT, E))
      return false;

    if (!this->emitCastBoolUint8(E))
      return false;
  }

  // We now have two bool values on the stack. Compare those.
  if (!this->emitAddUint8(E))
    return false;
  if (!this->emitConstUint8(2, E))
    return false;

  if (E->getOpcode() == BO_EQ) {
    if (!this->emitEQUint8(E))
      return false;
  } else if (E->getOpcode() == BO_NE) {
    if (!this->emitNEUint8(E))
      return false;
  } else
    return false;

  // In C, this returns an int.
  if (PrimType ResT = classifyPrim(E->getType()); ResT != PT_Bool)
    return this->emitCast(PT_Bool, ResT, E);
  return true;
}

/// Emits __real(SubExpr)
template <class Emitter>
bool Compiler<Emitter>::emitComplexReal(const Expr *SubExpr) {
  assert(SubExpr->getType()->isAnyComplexType());

  if (DiscardResult)
    return this->discard(SubExpr);

  if (!this->visit(SubExpr))
    return false;
  if (SubExpr->isLValue()) {
    if (!this->emitConstUint8(0, SubExpr))
      return false;
    return this->emitArrayElemPtrPopUint8(SubExpr);
  }

  // Rvalue, load the actual element.
  return this->emitArrayElemPop(classifyComplexElementType(SubExpr->getType()),
                                0, SubExpr);
}

template <class Emitter>
bool Compiler<Emitter>::emitComplexBoolCast(const Expr *E) {
  assert(!DiscardResult);
  PrimType ElemT = classifyComplexElementType(E->getType());
  // We emit the expression (__real(E) != 0 || __imag(E) != 0)
  // for us, that means (bool)E[0] || (bool)E[1]
  if (!this->emitArrayElem(ElemT, 0, E))
    return false;
  if (ElemT == PT_Float) {
    if (!this->emitCastFloatingIntegral(PT_Bool, E))
      return false;
  } else {
    if (!this->emitCast(ElemT, PT_Bool, E))
      return false;
  }

  // We now have the bool value of E[0] on the stack.
  LabelTy LabelTrue = this->getLabel();
  if (!this->jumpTrue(LabelTrue))
    return false;

  if (!this->emitArrayElemPop(ElemT, 1, E))
    return false;
  if (ElemT == PT_Float) {
    if (!this->emitCastFloatingIntegral(PT_Bool, E))
      return false;
  } else {
    if (!this->emitCast(ElemT, PT_Bool, E))
      return false;
  }
  // Leave the boolean value of E[1] on the stack.
  LabelTy EndLabel = this->getLabel();
  this->jump(EndLabel);

  this->emitLabel(LabelTrue);
  if (!this->emitPopPtr(E))
    return false;
  if (!this->emitConstBool(true, E))
    return false;

  this->fallthrough(EndLabel);
  this->emitLabel(EndLabel);

  return true;
}

template <class Emitter>
bool Compiler<Emitter>::VisitComplexUnaryOperator(const UnaryOperator *E) {
  const Expr *SubExpr = E->getSubExpr();
  assert(SubExpr->getType()->isAnyComplexType());

  if (DiscardResult)
    return this->discard(SubExpr);

  std::optional<PrimType> ResT = classify(E);
  auto prepareResult = [=]() -> bool {
    if (!ResT && !Initializing) {
      std::optional<unsigned> LocalIndex = allocateLocal(SubExpr);
      if (!LocalIndex)
        return false;
      return this->emitGetPtrLocal(*LocalIndex, E);
    }

    return true;
  };

  // The offset of the temporary, if we created one.
  unsigned SubExprOffset = ~0u;
  auto createTemp = [=, &SubExprOffset]() -> bool {
    SubExprOffset = this->allocateLocalPrimitive(SubExpr, PT_Ptr, true, false);
    if (!this->visit(SubExpr))
      return false;
    return this->emitSetLocal(PT_Ptr, SubExprOffset, E);
  };

  PrimType ElemT = classifyComplexElementType(SubExpr->getType());
  auto getElem = [=](unsigned Offset, unsigned Index) -> bool {
    if (!this->emitGetLocal(PT_Ptr, Offset, E))
      return false;
    return this->emitArrayElemPop(ElemT, Index, E);
  };

  switch (E->getOpcode()) {
  case UO_Minus:
    if (!prepareResult())
      return false;
    if (!createTemp())
      return false;
    for (unsigned I = 0; I != 2; ++I) {
      if (!getElem(SubExprOffset, I))
        return false;
      if (!this->emitNeg(ElemT, E))
        return false;
      if (!this->emitInitElem(ElemT, I, E))
        return false;
    }
    break;

  case UO_Plus:   // +x
  case UO_AddrOf: // &x
  case UO_Deref:  // *x
    return this->delegate(SubExpr);

  case UO_LNot:
    if (!this->visit(SubExpr))
      return false;
    if (!this->emitComplexBoolCast(SubExpr))
      return false;
    if (!this->emitInvBool(E))
      return false;
    if (PrimType ET = classifyPrim(E->getType()); ET != PT_Bool)
      return this->emitCast(PT_Bool, ET, E);
    return true;

  case UO_Real:
    return this->emitComplexReal(SubExpr);

  case UO_Imag:
    if (!this->visit(SubExpr))
      return false;

    if (SubExpr->isLValue()) {
      if (!this->emitConstUint8(1, E))
        return false;
      return this->emitArrayElemPtrPopUint8(E);
    }

    // Since our _Complex implementation does not map to a primitive type,
    // we sometimes have to do the lvalue-to-rvalue conversion here manually.
    return this->emitArrayElemPop(classifyPrim(E->getType()), 1, E);

  case UO_Not: // ~x
    if (!this->visit(SubExpr))
      return false;
    // Negate the imaginary component.
    if (!this->emitArrayElem(ElemT, 1, E))
      return false;
    if (!this->emitNeg(ElemT, E))
      return false;
    if (!this->emitInitElem(ElemT, 1, E))
      return false;
    return DiscardResult ? this->emitPopPtr(E) : true;

  case UO_Extension:
    return this->delegate(SubExpr);

  default:
    return this->emitInvalid(E);
  }

  return true;
}

namespace clang {
namespace interp {

template class Compiler<ByteCodeEmitter>;
template class Compiler<EvalEmitter>;

} // namespace interp
} // namespace clang
