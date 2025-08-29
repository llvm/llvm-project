//===--- InterpBuiltin.cpp - Interpreter for the constexpr VM ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../ExprConstShared.h"
#include "Boolean.h"
#include "EvalEmitter.h"
#include "Interp.h"
#include "InterpBuiltinBitCast.h"
#include "PrimType.h"
#include "clang/AST/OSLog.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SipHash.h"

namespace clang {
namespace interp {

LLVM_ATTRIBUTE_UNUSED static bool isNoopBuiltin(unsigned ID) {
  switch (ID) {
  case Builtin::BIas_const:
  case Builtin::BIforward:
  case Builtin::BIforward_like:
  case Builtin::BImove:
  case Builtin::BImove_if_noexcept:
  case Builtin::BIaddressof:
  case Builtin::BI__addressof:
  case Builtin::BI__builtin_addressof:
  case Builtin::BI__builtin_launder:
    return true;
  default:
    return false;
  }
  return false;
}

static void discard(InterpStack &Stk, PrimType T) {
  TYPE_SWITCH(T, { Stk.discard<T>(); });
}

static APSInt popToAPSInt(InterpStack &Stk, PrimType T) {
  INT_TYPE_SWITCH(T, return Stk.pop<T>().toAPSInt());
}

/// Pushes \p Val on the stack as the type given by \p QT.
static void pushInteger(InterpState &S, const APSInt &Val, QualType QT) {
  assert(QT->isSignedIntegerOrEnumerationType() ||
         QT->isUnsignedIntegerOrEnumerationType());
  OptPrimType T = S.getContext().classify(QT);
  assert(T);

  unsigned BitWidth = S.getASTContext().getTypeSize(QT);

  if (T == PT_IntAPS) {
    auto Result = S.allocAP<IntegralAP<true>>(BitWidth);
    Result.copy(Val);
    S.Stk.push<IntegralAP<true>>(Result);
    return;
  }

  if (T == PT_IntAP) {
    auto Result = S.allocAP<IntegralAP<false>>(BitWidth);
    Result.copy(Val);
    S.Stk.push<IntegralAP<false>>(Result);
    return;
  }

  if (QT->isSignedIntegerOrEnumerationType()) {
    int64_t V = Val.getSExtValue();
    INT_TYPE_SWITCH(*T, { S.Stk.push<T>(T::from(V, BitWidth)); });
  } else {
    assert(QT->isUnsignedIntegerOrEnumerationType());
    uint64_t V = Val.getZExtValue();
    INT_TYPE_SWITCH(*T, { S.Stk.push<T>(T::from(V, BitWidth)); });
  }
}

template <typename T>
static void pushInteger(InterpState &S, T Val, QualType QT) {
  if constexpr (std::is_same_v<T, APInt>)
    pushInteger(S, APSInt(Val, !std::is_signed_v<T>), QT);
  else if constexpr (std::is_same_v<T, APSInt>)
    pushInteger(S, Val, QT);
  else
    pushInteger(S,
                APSInt(APInt(sizeof(T) * 8, static_cast<uint64_t>(Val),
                             std::is_signed_v<T>),
                       !std::is_signed_v<T>),
                QT);
}

static void assignInteger(InterpState &S, const Pointer &Dest, PrimType ValueT,
                          const APSInt &Value) {

  if (ValueT == PT_IntAPS) {
    Dest.deref<IntegralAP<true>>() =
        S.allocAP<IntegralAP<true>>(Value.getBitWidth());
    Dest.deref<IntegralAP<true>>().copy(Value);
  } else if (ValueT == PT_IntAP) {
    Dest.deref<IntegralAP<false>>() =
        S.allocAP<IntegralAP<false>>(Value.getBitWidth());
    Dest.deref<IntegralAP<false>>().copy(Value);
  } else {
    INT_TYPE_SWITCH_NO_BOOL(
        ValueT, { Dest.deref<T>() = T::from(static_cast<T>(Value)); });
  }
}

static QualType getElemType(const Pointer &P) {
  const Descriptor *Desc = P.getFieldDesc();
  QualType T = Desc->getType();
  if (Desc->isPrimitive())
    return T;
  if (T->isPointerType())
    return T->getAs<PointerType>()->getPointeeType();
  if (Desc->isArray())
    return Desc->getElemQualType();
  if (const auto *AT = T->getAsArrayTypeUnsafe())
    return AT->getElementType();
  return T;
}

static void diagnoseNonConstexprBuiltin(InterpState &S, CodePtr OpPC,
                                        unsigned ID) {
  if (!S.diagnosing())
    return;

  auto Loc = S.Current->getSource(OpPC);
  if (S.getLangOpts().CPlusPlus11)
    S.CCEDiag(Loc, diag::note_constexpr_invalid_function)
        << /*isConstexpr=*/0 << /*isConstructor=*/0
        << S.getASTContext().BuiltinInfo.getQuotedName(ID);
  else
    S.CCEDiag(Loc, diag::note_invalid_subexpr_in_const_expr);
}

static llvm::APSInt convertBoolVectorToInt(const Pointer &Val) {
  assert(Val.getFieldDesc()->isPrimitiveArray() &&
         Val.getFieldDesc()->getElemQualType()->isBooleanType() &&
         "Not a boolean vector");
  unsigned NumElems = Val.getNumElems();

  // Each element is one bit, so create an integer with NumElts bits.
  llvm::APSInt Result(NumElems, 0);
  for (unsigned I = 0; I != NumElems; ++I) {
    if (Val.elem<bool>(I))
      Result.setBit(I);
  }

  return Result;
}

static bool interp__builtin_is_constant_evaluated(InterpState &S, CodePtr OpPC,
                                                  const InterpFrame *Frame,
                                                  const CallExpr *Call) {
  unsigned Depth = S.Current->getDepth();
  auto isStdCall = [](const FunctionDecl *F) -> bool {
    return F && F->isInStdNamespace() && F->getIdentifier() &&
           F->getIdentifier()->isStr("is_constant_evaluated");
  };
  const InterpFrame *Caller = Frame->Caller;
  // The current frame is the one for __builtin_is_constant_evaluated.
  // The one above that, potentially the one for std::is_constant_evaluated().
  if (S.inConstantContext() && !S.checkingPotentialConstantExpression() &&
      S.getEvalStatus().Diag &&
      (Depth == 0 || (Depth == 1 && isStdCall(Frame->getCallee())))) {
    if (Caller && isStdCall(Frame->getCallee())) {
      const Expr *E = Caller->getExpr(Caller->getRetPC());
      S.report(E->getExprLoc(),
               diag::warn_is_constant_evaluated_always_true_constexpr)
          << "std::is_constant_evaluated" << E->getSourceRange();
    } else {
      S.report(Call->getExprLoc(),
               diag::warn_is_constant_evaluated_always_true_constexpr)
          << "__builtin_is_constant_evaluated" << Call->getSourceRange();
    }
  }

  S.Stk.push<Boolean>(Boolean::from(S.inConstantContext()));
  return true;
}

// __builtin_assume(int)
static bool interp__builtin_assume(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const CallExpr *Call) {
  assert(Call->getNumArgs() == 1);
  discard(S.Stk, *S.getContext().classify(Call->getArg(0)));
  return true;
}

static bool interp__builtin_strcmp(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const CallExpr *Call, unsigned ID) {
  uint64_t Limit = ~static_cast<uint64_t>(0);
  if (ID == Builtin::BIstrncmp || ID == Builtin::BI__builtin_strncmp ||
      ID == Builtin::BIwcsncmp || ID == Builtin::BI__builtin_wcsncmp)
    Limit = popToAPSInt(S.Stk, *S.getContext().classify(Call->getArg(2)))
                .getZExtValue();

  const Pointer &B = S.Stk.pop<Pointer>();
  const Pointer &A = S.Stk.pop<Pointer>();
  if (ID == Builtin::BIstrcmp || ID == Builtin::BIstrncmp ||
      ID == Builtin::BIwcscmp || ID == Builtin::BIwcsncmp)
    diagnoseNonConstexprBuiltin(S, OpPC, ID);

  if (Limit == 0) {
    pushInteger(S, 0, Call->getType());
    return true;
  }

  if (!CheckLive(S, OpPC, A, AK_Read) || !CheckLive(S, OpPC, B, AK_Read))
    return false;

  if (A.isDummy() || B.isDummy())
    return false;
  if (!A.isBlockPointer() || !B.isBlockPointer())
    return false;

  bool IsWide = ID == Builtin::BIwcscmp || ID == Builtin::BIwcsncmp ||
                ID == Builtin::BI__builtin_wcscmp ||
                ID == Builtin::BI__builtin_wcsncmp;
  assert(A.getFieldDesc()->isPrimitiveArray());
  assert(B.getFieldDesc()->isPrimitiveArray());

  // Different element types shouldn't happen, but with casts they can.
  if (!S.getASTContext().hasSameUnqualifiedType(getElemType(A), getElemType(B)))
    return false;

  PrimType ElemT = *S.getContext().classify(getElemType(A));

  auto returnResult = [&](int V) -> bool {
    pushInteger(S, V, Call->getType());
    return true;
  };

  unsigned IndexA = A.getIndex();
  unsigned IndexB = B.getIndex();
  uint64_t Steps = 0;
  for (;; ++IndexA, ++IndexB, ++Steps) {

    if (Steps >= Limit)
      break;
    const Pointer &PA = A.atIndex(IndexA);
    const Pointer &PB = B.atIndex(IndexB);
    if (!CheckRange(S, OpPC, PA, AK_Read) ||
        !CheckRange(S, OpPC, PB, AK_Read)) {
      return false;
    }

    if (IsWide) {
      INT_TYPE_SWITCH(ElemT, {
        T CA = PA.deref<T>();
        T CB = PB.deref<T>();
        if (CA > CB)
          return returnResult(1);
        if (CA < CB)
          return returnResult(-1);
        if (CA.isZero() || CB.isZero())
          return returnResult(0);
      });
      continue;
    }

    uint8_t CA = PA.deref<uint8_t>();
    uint8_t CB = PB.deref<uint8_t>();

    if (CA > CB)
      return returnResult(1);
    if (CA < CB)
      return returnResult(-1);
    if (CA == 0 || CB == 0)
      return returnResult(0);
  }

  return returnResult(0);
}

static bool interp__builtin_strlen(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const CallExpr *Call, unsigned ID) {
  const Pointer &StrPtr = S.Stk.pop<Pointer>();

  if (ID == Builtin::BIstrlen || ID == Builtin::BIwcslen)
    diagnoseNonConstexprBuiltin(S, OpPC, ID);

  if (!CheckArray(S, OpPC, StrPtr))
    return false;

  if (!CheckLive(S, OpPC, StrPtr, AK_Read))
    return false;

  if (!CheckDummy(S, OpPC, StrPtr.block(), AK_Read))
    return false;

  assert(StrPtr.getFieldDesc()->isPrimitiveArray());
  unsigned ElemSize = StrPtr.getFieldDesc()->getElemSize();

  if (ID == Builtin::BI__builtin_wcslen || ID == Builtin::BIwcslen) {
    [[maybe_unused]] const ASTContext &AC = S.getASTContext();
    assert(ElemSize == AC.getTypeSizeInChars(AC.getWCharType()).getQuantity());
  }

  size_t Len = 0;
  for (size_t I = StrPtr.getIndex();; ++I, ++Len) {
    const Pointer &ElemPtr = StrPtr.atIndex(I);

    if (!CheckRange(S, OpPC, ElemPtr, AK_Read))
      return false;

    uint32_t Val;
    switch (ElemSize) {
    case 1:
      Val = ElemPtr.deref<uint8_t>();
      break;
    case 2:
      Val = ElemPtr.deref<uint16_t>();
      break;
    case 4:
      Val = ElemPtr.deref<uint32_t>();
      break;
    default:
      llvm_unreachable("Unsupported char size");
    }
    if (Val == 0)
      break;
  }

  pushInteger(S, Len, Call->getType());

  return true;
}

static bool interp__builtin_nan(InterpState &S, CodePtr OpPC,
                                const InterpFrame *Frame, const CallExpr *Call,
                                bool Signaling) {
  const Pointer &Arg = S.Stk.pop<Pointer>();

  if (!CheckLoad(S, OpPC, Arg))
    return false;

  assert(Arg.getFieldDesc()->isPrimitiveArray());

  // Convert the given string to an integer using StringRef's API.
  llvm::APInt Fill;
  std::string Str;
  assert(Arg.getNumElems() >= 1);
  for (unsigned I = 0;; ++I) {
    const Pointer &Elem = Arg.atIndex(I);

    if (!CheckLoad(S, OpPC, Elem))
      return false;

    if (Elem.deref<int8_t>() == 0)
      break;

    Str += Elem.deref<char>();
  }

  // Treat empty strings as if they were zero.
  if (Str.empty())
    Fill = llvm::APInt(32, 0);
  else if (StringRef(Str).getAsInteger(0, Fill))
    return false;

  const llvm::fltSemantics &TargetSemantics =
      S.getASTContext().getFloatTypeSemantics(
          Call->getDirectCallee()->getReturnType());

  Floating Result = S.allocFloat(TargetSemantics);
  if (S.getASTContext().getTargetInfo().isNan2008()) {
    if (Signaling)
      Result.copy(
          llvm::APFloat::getSNaN(TargetSemantics, /*Negative=*/false, &Fill));
    else
      Result.copy(
          llvm::APFloat::getQNaN(TargetSemantics, /*Negative=*/false, &Fill));
  } else {
    // Prior to IEEE 754-2008, architectures were allowed to choose whether
    // the first bit of their significand was set for qNaN or sNaN. MIPS chose
    // a different encoding to what became a standard in 2008, and for pre-
    // 2008 revisions, MIPS interpreted sNaN-2008 as qNan and qNaN-2008 as
    // sNaN. This is now known as "legacy NaN" encoding.
    if (Signaling)
      Result.copy(
          llvm::APFloat::getQNaN(TargetSemantics, /*Negative=*/false, &Fill));
    else
      Result.copy(
          llvm::APFloat::getSNaN(TargetSemantics, /*Negative=*/false, &Fill));
  }

  S.Stk.push<Floating>(Result);
  return true;
}

static bool interp__builtin_inf(InterpState &S, CodePtr OpPC,
                                const InterpFrame *Frame,
                                const CallExpr *Call) {
  const llvm::fltSemantics &TargetSemantics =
      S.getASTContext().getFloatTypeSemantics(
          Call->getDirectCallee()->getReturnType());

  Floating Result = S.allocFloat(TargetSemantics);
  Result.copy(APFloat::getInf(TargetSemantics));
  S.Stk.push<Floating>(Result);
  return true;
}

static bool interp__builtin_copysign(InterpState &S, CodePtr OpPC,
                                     const InterpFrame *Frame) {
  const Floating &Arg2 = S.Stk.pop<Floating>();
  const Floating &Arg1 = S.Stk.pop<Floating>();
  Floating Result = S.allocFloat(Arg1.getSemantics());

  APFloat Copy = Arg1.getAPFloat();
  Copy.copySign(Arg2.getAPFloat());
  Result.copy(Copy);
  S.Stk.push<Floating>(Result);

  return true;
}

static bool interp__builtin_fmin(InterpState &S, CodePtr OpPC,
                                 const InterpFrame *Frame, bool IsNumBuiltin) {
  const Floating &RHS = S.Stk.pop<Floating>();
  const Floating &LHS = S.Stk.pop<Floating>();
  Floating Result = S.allocFloat(LHS.getSemantics());

  if (IsNumBuiltin)
    Result.copy(llvm::minimumnum(LHS.getAPFloat(), RHS.getAPFloat()));
  else
    Result.copy(minnum(LHS.getAPFloat(), RHS.getAPFloat()));
  S.Stk.push<Floating>(Result);
  return true;
}

static bool interp__builtin_fmax(InterpState &S, CodePtr OpPC,
                                 const InterpFrame *Frame, bool IsNumBuiltin) {
  const Floating &RHS = S.Stk.pop<Floating>();
  const Floating &LHS = S.Stk.pop<Floating>();
  Floating Result = S.allocFloat(LHS.getSemantics());

  if (IsNumBuiltin)
    Result.copy(llvm::maximumnum(LHS.getAPFloat(), RHS.getAPFloat()));
  else
    Result.copy(maxnum(LHS.getAPFloat(), RHS.getAPFloat()));
  S.Stk.push<Floating>(Result);
  return true;
}

/// Defined as __builtin_isnan(...), to accommodate the fact that it can
/// take a float, double, long double, etc.
/// But for us, that's all a Floating anyway.
static bool interp__builtin_isnan(InterpState &S, CodePtr OpPC,
                                  const InterpFrame *Frame,
                                  const CallExpr *Call) {
  const Floating &Arg = S.Stk.pop<Floating>();

  pushInteger(S, Arg.isNan(), Call->getType());
  return true;
}

static bool interp__builtin_issignaling(InterpState &S, CodePtr OpPC,
                                        const InterpFrame *Frame,
                                        const CallExpr *Call) {
  const Floating &Arg = S.Stk.pop<Floating>();

  pushInteger(S, Arg.isSignaling(), Call->getType());
  return true;
}

static bool interp__builtin_isinf(InterpState &S, CodePtr OpPC,
                                  const InterpFrame *Frame, bool CheckSign,
                                  const CallExpr *Call) {
  const Floating &Arg = S.Stk.pop<Floating>();
  APFloat F = Arg.getAPFloat();
  bool IsInf = F.isInfinity();

  if (CheckSign)
    pushInteger(S, IsInf ? (F.isNegative() ? -1 : 1) : 0, Call->getType());
  else
    pushInteger(S, IsInf, Call->getType());
  return true;
}

static bool interp__builtin_isfinite(InterpState &S, CodePtr OpPC,
                                     const InterpFrame *Frame,
                                     const CallExpr *Call) {
  const Floating &Arg = S.Stk.pop<Floating>();

  pushInteger(S, Arg.isFinite(), Call->getType());
  return true;
}

static bool interp__builtin_isnormal(InterpState &S, CodePtr OpPC,
                                     const InterpFrame *Frame,
                                     const CallExpr *Call) {
  const Floating &Arg = S.Stk.pop<Floating>();

  pushInteger(S, Arg.isNormal(), Call->getType());
  return true;
}

static bool interp__builtin_issubnormal(InterpState &S, CodePtr OpPC,
                                        const InterpFrame *Frame,
                                        const CallExpr *Call) {
  const Floating &Arg = S.Stk.pop<Floating>();

  pushInteger(S, Arg.isDenormal(), Call->getType());
  return true;
}

static bool interp__builtin_iszero(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const CallExpr *Call) {
  const Floating &Arg = S.Stk.pop<Floating>();

  pushInteger(S, Arg.isZero(), Call->getType());
  return true;
}

static bool interp__builtin_signbit(InterpState &S, CodePtr OpPC,
                                    const InterpFrame *Frame,
                                    const CallExpr *Call) {
  const Floating &Arg = S.Stk.pop<Floating>();

  pushInteger(S, Arg.isNegative(), Call->getType());
  return true;
}

static bool interp_floating_comparison(InterpState &S, CodePtr OpPC,
                                       const CallExpr *Call, unsigned ID) {
  const Floating &RHS = S.Stk.pop<Floating>();
  const Floating &LHS = S.Stk.pop<Floating>();

  pushInteger(
      S,
      [&] {
        switch (ID) {
        case Builtin::BI__builtin_isgreater:
          return LHS > RHS;
        case Builtin::BI__builtin_isgreaterequal:
          return LHS >= RHS;
        case Builtin::BI__builtin_isless:
          return LHS < RHS;
        case Builtin::BI__builtin_islessequal:
          return LHS <= RHS;
        case Builtin::BI__builtin_islessgreater: {
          ComparisonCategoryResult cmp = LHS.compare(RHS);
          return cmp == ComparisonCategoryResult::Less ||
                 cmp == ComparisonCategoryResult::Greater;
        }
        case Builtin::BI__builtin_isunordered:
          return LHS.compare(RHS) == ComparisonCategoryResult::Unordered;
        default:
          llvm_unreachable("Unexpected builtin ID: Should be a floating point "
                           "comparison function");
        }
      }(),
      Call->getType());
  return true;
}

/// First parameter to __builtin_isfpclass is the floating value, the
/// second one is an integral value.
static bool interp__builtin_isfpclass(InterpState &S, CodePtr OpPC,
                                      const InterpFrame *Frame,
                                      const CallExpr *Call) {
  PrimType FPClassArgT = *S.getContext().classify(Call->getArg(1)->getType());
  APSInt FPClassArg = popToAPSInt(S.Stk, FPClassArgT);
  const Floating &F = S.Stk.pop<Floating>();

  int32_t Result = static_cast<int32_t>(
      (F.classify() & std::move(FPClassArg)).getZExtValue());
  pushInteger(S, Result, Call->getType());

  return true;
}

/// Five int values followed by one floating value.
/// __builtin_fpclassify(int, int, int, int, int, float)
static bool interp__builtin_fpclassify(InterpState &S, CodePtr OpPC,
                                       const InterpFrame *Frame,
                                       const CallExpr *Call) {
  const Floating &Val = S.Stk.pop<Floating>();

  PrimType IntT = *S.getContext().classify(Call->getArg(0));
  APSInt Values[5];
  for (unsigned I = 0; I != 5; ++I)
    Values[4 - I] = popToAPSInt(S.Stk, IntT);

  unsigned Index;
  switch (Val.getCategory()) {
  case APFloat::fcNaN:
    Index = 0;
    break;
  case APFloat::fcInfinity:
    Index = 1;
    break;
  case APFloat::fcNormal:
    Index = Val.isDenormal() ? 3 : 2;
    break;
  case APFloat::fcZero:
    Index = 4;
    break;
  }

  // The last argument is first on the stack.
  assert(Index <= 4);

  pushInteger(S, Values[Index], Call->getType());
  return true;
}

static inline Floating abs(InterpState &S, const Floating &In) {
  if (!In.isNegative())
    return In;

  Floating Output = S.allocFloat(In.getSemantics());
  APFloat New = In.getAPFloat();
  New.changeSign();
  Output.copy(New);
  return Output;
}

// The C standard says "fabs raises no floating-point exceptions,
// even if x is a signaling NaN. The returned value is independent of
// the current rounding direction mode."  Therefore constant folding can
// proceed without regard to the floating point settings.
// Reference, WG14 N2478 F.10.4.3
static bool interp__builtin_fabs(InterpState &S, CodePtr OpPC,
                                 const InterpFrame *Frame) {
  const Floating &Val = S.Stk.pop<Floating>();
  S.Stk.push<Floating>(abs(S, Val));
  return true;
}

static bool interp__builtin_abs(InterpState &S, CodePtr OpPC,
                                const InterpFrame *Frame,
                                const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Val = popToAPSInt(S.Stk, ArgT);
  if (Val ==
      APSInt(APInt::getSignedMinValue(Val.getBitWidth()), /*IsUnsigned=*/false))
    return false;
  if (Val.isNegative())
    Val.negate();
  pushInteger(S, Val, Call->getType());
  return true;
}

static bool interp__builtin_popcount(InterpState &S, CodePtr OpPC,
                                     const InterpFrame *Frame,
                                     const CallExpr *Call) {
  APSInt Val;
  if (Call->getArg(0)->getType()->isExtVectorBoolType()) {
    const Pointer &Arg = S.Stk.pop<Pointer>();
    Val = convertBoolVectorToInt(Arg);
  } else {
    PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
    Val = popToAPSInt(S.Stk, ArgT);
  }
  pushInteger(S, Val.popcount(), Call->getType());
  return true;
}

static bool interp__builtin_parity(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Val = popToAPSInt(S.Stk, ArgT);
  pushInteger(S, Val.popcount() % 2, Call->getType());
  return true;
}

static bool interp__builtin_clrsb(InterpState &S, CodePtr OpPC,
                                  const InterpFrame *Frame,
                                  const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Val = popToAPSInt(S.Stk, ArgT);
  pushInteger(S, Val.getBitWidth() - Val.getSignificantBits(), Call->getType());
  return true;
}

static bool interp__builtin_bitreverse(InterpState &S, CodePtr OpPC,
                                       const InterpFrame *Frame,
                                       const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Val = popToAPSInt(S.Stk, ArgT);
  pushInteger(S, Val.reverseBits(), Call->getType());
  return true;
}

static bool interp__builtin_classify_type(InterpState &S, CodePtr OpPC,
                                          const InterpFrame *Frame,
                                          const CallExpr *Call) {
  // This is an unevaluated call, so there are no arguments on the stack.
  assert(Call->getNumArgs() == 1);
  const Expr *Arg = Call->getArg(0);

  GCCTypeClass ResultClass =
      EvaluateBuiltinClassifyType(Arg->getType(), S.getLangOpts());
  int32_t ReturnVal = static_cast<int32_t>(ResultClass);
  pushInteger(S, ReturnVal, Call->getType());
  return true;
}

// __builtin_expect(long, long)
// __builtin_expect_with_probability(long, long, double)
static bool interp__builtin_expect(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const CallExpr *Call) {
  // The return value is simply the value of the first parameter.
  // We ignore the probability.
  unsigned NumArgs = Call->getNumArgs();
  assert(NumArgs == 2 || NumArgs == 3);

  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  if (NumArgs == 3)
    S.Stk.discard<Floating>();
  discard(S.Stk, ArgT);

  APSInt Val = popToAPSInt(S.Stk, ArgT);
  pushInteger(S, Val, Call->getType());
  return true;
}

/// rotateleft(value, amount)
static bool interp__builtin_rotate(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const CallExpr *Call, bool Right) {
  PrimType AmountT = *S.getContext().classify(Call->getArg(1)->getType());
  PrimType ValueT = *S.getContext().classify(Call->getArg(0)->getType());

  APSInt Amount = popToAPSInt(S.Stk, AmountT);
  APSInt Value = popToAPSInt(S.Stk, ValueT);

  APSInt Result;
  if (Right)
    Result = APSInt(Value.rotr(Amount.urem(Value.getBitWidth())),
                    /*IsUnsigned=*/true);
  else // Left.
    Result = APSInt(Value.rotl(Amount.urem(Value.getBitWidth())),
                    /*IsUnsigned=*/true);

  pushInteger(S, Result, Call->getType());
  return true;
}

static bool interp__builtin_ffs(InterpState &S, CodePtr OpPC,
                                const InterpFrame *Frame,
                                const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Value = popToAPSInt(S.Stk, ArgT);

  uint64_t N = Value.countr_zero();
  pushInteger(S, N == Value.getBitWidth() ? 0 : N + 1, Call->getType());
  return true;
}

static bool interp__builtin_addressof(InterpState &S, CodePtr OpPC,
                                      const InterpFrame *Frame,
                                      const CallExpr *Call) {
#ifndef NDEBUG
  assert(Call->getArg(0)->isLValue());
  PrimType PtrT = S.getContext().classify(Call->getArg(0)).value_or(PT_Ptr);
  assert(PtrT == PT_Ptr &&
         "Unsupported pointer type passed to __builtin_addressof()");
#endif
  return true;
}

static bool interp__builtin_move(InterpState &S, CodePtr OpPC,
                                 const InterpFrame *Frame,
                                 const CallExpr *Call) {
  return Call->getDirectCallee()->isConstexpr();
}

static bool interp__builtin_eh_return_data_regno(InterpState &S, CodePtr OpPC,
                                                 const InterpFrame *Frame,
                                                 const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Arg = popToAPSInt(S.Stk, ArgT);

  int Result = S.getASTContext().getTargetInfo().getEHDataRegisterNumber(
      Arg.getZExtValue());
  pushInteger(S, Result, Call->getType());
  return true;
}

// Two integral values followed by a pointer (lhs, rhs, resultOut)
static bool interp__builtin_overflowop(InterpState &S, CodePtr OpPC,
                                       const CallExpr *Call,
                                       unsigned BuiltinOp) {
  const Pointer &ResultPtr = S.Stk.pop<Pointer>();
  if (ResultPtr.isDummy())
    return false;

  PrimType RHST = *S.getContext().classify(Call->getArg(1)->getType());
  PrimType LHST = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt RHS = popToAPSInt(S.Stk, RHST);
  APSInt LHS = popToAPSInt(S.Stk, LHST);
  QualType ResultType = Call->getArg(2)->getType()->getPointeeType();
  PrimType ResultT = *S.getContext().classify(ResultType);
  bool Overflow;

  APSInt Result;
  if (BuiltinOp == Builtin::BI__builtin_add_overflow ||
      BuiltinOp == Builtin::BI__builtin_sub_overflow ||
      BuiltinOp == Builtin::BI__builtin_mul_overflow) {
    bool IsSigned = LHS.isSigned() || RHS.isSigned() ||
                    ResultType->isSignedIntegerOrEnumerationType();
    bool AllSigned = LHS.isSigned() && RHS.isSigned() &&
                     ResultType->isSignedIntegerOrEnumerationType();
    uint64_t LHSSize = LHS.getBitWidth();
    uint64_t RHSSize = RHS.getBitWidth();
    uint64_t ResultSize = S.getASTContext().getTypeSize(ResultType);
    uint64_t MaxBits = std::max(std::max(LHSSize, RHSSize), ResultSize);

    // Add an additional bit if the signedness isn't uniformly agreed to. We
    // could do this ONLY if there is a signed and an unsigned that both have
    // MaxBits, but the code to check that is pretty nasty.  The issue will be
    // caught in the shrink-to-result later anyway.
    if (IsSigned && !AllSigned)
      ++MaxBits;

    LHS = APSInt(LHS.extOrTrunc(MaxBits), !IsSigned);
    RHS = APSInt(RHS.extOrTrunc(MaxBits), !IsSigned);
    Result = APSInt(MaxBits, !IsSigned);
  }

  // Find largest int.
  switch (BuiltinOp) {
  default:
    llvm_unreachable("Invalid value for BuiltinOp");
  case Builtin::BI__builtin_add_overflow:
  case Builtin::BI__builtin_sadd_overflow:
  case Builtin::BI__builtin_saddl_overflow:
  case Builtin::BI__builtin_saddll_overflow:
  case Builtin::BI__builtin_uadd_overflow:
  case Builtin::BI__builtin_uaddl_overflow:
  case Builtin::BI__builtin_uaddll_overflow:
    Result = LHS.isSigned() ? LHS.sadd_ov(RHS, Overflow)
                            : LHS.uadd_ov(RHS, Overflow);
    break;
  case Builtin::BI__builtin_sub_overflow:
  case Builtin::BI__builtin_ssub_overflow:
  case Builtin::BI__builtin_ssubl_overflow:
  case Builtin::BI__builtin_ssubll_overflow:
  case Builtin::BI__builtin_usub_overflow:
  case Builtin::BI__builtin_usubl_overflow:
  case Builtin::BI__builtin_usubll_overflow:
    Result = LHS.isSigned() ? LHS.ssub_ov(RHS, Overflow)
                            : LHS.usub_ov(RHS, Overflow);
    break;
  case Builtin::BI__builtin_mul_overflow:
  case Builtin::BI__builtin_smul_overflow:
  case Builtin::BI__builtin_smull_overflow:
  case Builtin::BI__builtin_smulll_overflow:
  case Builtin::BI__builtin_umul_overflow:
  case Builtin::BI__builtin_umull_overflow:
  case Builtin::BI__builtin_umulll_overflow:
    Result = LHS.isSigned() ? LHS.smul_ov(RHS, Overflow)
                            : LHS.umul_ov(RHS, Overflow);
    break;
  }

  // In the case where multiple sizes are allowed, truncate and see if
  // the values are the same.
  if (BuiltinOp == Builtin::BI__builtin_add_overflow ||
      BuiltinOp == Builtin::BI__builtin_sub_overflow ||
      BuiltinOp == Builtin::BI__builtin_mul_overflow) {
    // APSInt doesn't have a TruncOrSelf, so we use extOrTrunc instead,
    // since it will give us the behavior of a TruncOrSelf in the case where
    // its parameter <= its size.  We previously set Result to be at least the
    // type-size of the result, so getTypeSize(ResultType) <= Resu
    APSInt Temp = Result.extOrTrunc(S.getASTContext().getTypeSize(ResultType));
    Temp.setIsSigned(ResultType->isSignedIntegerOrEnumerationType());

    if (!APSInt::isSameValue(Temp, Result))
      Overflow = true;
    Result = std::move(Temp);
  }

  // Write Result to ResultPtr and put Overflow on the stack.
  assignInteger(S, ResultPtr, ResultT, Result);
  if (ResultPtr.canBeInitialized())
    ResultPtr.initialize();

  assert(Call->getDirectCallee()->getReturnType()->isBooleanType());
  S.Stk.push<Boolean>(Overflow);
  return true;
}

/// Three integral values followed by a pointer (lhs, rhs, carry, carryOut).
static bool interp__builtin_carryop(InterpState &S, CodePtr OpPC,
                                    const InterpFrame *Frame,
                                    const CallExpr *Call, unsigned BuiltinOp) {
  const Pointer &CarryOutPtr = S.Stk.pop<Pointer>();
  PrimType LHST = *S.getContext().classify(Call->getArg(0)->getType());
  PrimType RHST = *S.getContext().classify(Call->getArg(1)->getType());
  APSInt CarryIn = popToAPSInt(S.Stk, LHST);
  APSInt RHS = popToAPSInt(S.Stk, RHST);
  APSInt LHS = popToAPSInt(S.Stk, LHST);

  APSInt CarryOut;

  APSInt Result;
  // Copy the number of bits and sign.
  Result = LHS;
  CarryOut = LHS;

  bool FirstOverflowed = false;
  bool SecondOverflowed = false;
  switch (BuiltinOp) {
  default:
    llvm_unreachable("Invalid value for BuiltinOp");
  case Builtin::BI__builtin_addcb:
  case Builtin::BI__builtin_addcs:
  case Builtin::BI__builtin_addc:
  case Builtin::BI__builtin_addcl:
  case Builtin::BI__builtin_addcll:
    Result =
        LHS.uadd_ov(RHS, FirstOverflowed).uadd_ov(CarryIn, SecondOverflowed);
    break;
  case Builtin::BI__builtin_subcb:
  case Builtin::BI__builtin_subcs:
  case Builtin::BI__builtin_subc:
  case Builtin::BI__builtin_subcl:
  case Builtin::BI__builtin_subcll:
    Result =
        LHS.usub_ov(RHS, FirstOverflowed).usub_ov(CarryIn, SecondOverflowed);
    break;
  }
  // It is possible for both overflows to happen but CGBuiltin uses an OR so
  // this is consistent.
  CarryOut = (uint64_t)(FirstOverflowed | SecondOverflowed);

  QualType CarryOutType = Call->getArg(3)->getType()->getPointeeType();
  PrimType CarryOutT = *S.getContext().classify(CarryOutType);
  assignInteger(S, CarryOutPtr, CarryOutT, CarryOut);
  CarryOutPtr.initialize();

  assert(Call->getType() == Call->getArg(0)->getType());
  pushInteger(S, Result, Call->getType());
  return true;
}

static bool interp__builtin_clz(InterpState &S, CodePtr OpPC,
                                const InterpFrame *Frame, const CallExpr *Call,
                                unsigned BuiltinOp) {

  std::optional<APSInt> Fallback;
  if (BuiltinOp == Builtin::BI__builtin_clzg && Call->getNumArgs() == 2) {
    PrimType FallbackT = *S.getContext().classify(Call->getArg(1));
    Fallback = popToAPSInt(S.Stk, FallbackT);
  }
  APSInt Val;
  if (Call->getArg(0)->getType()->isExtVectorBoolType()) {
    const Pointer &Arg = S.Stk.pop<Pointer>();
    Val = convertBoolVectorToInt(Arg);
  } else {
    PrimType ValT = *S.getContext().classify(Call->getArg(0));
    Val = popToAPSInt(S.Stk, ValT);
  }

  // When the argument is 0, the result of GCC builtins is undefined, whereas
  // for Microsoft intrinsics, the result is the bit-width of the argument.
  bool ZeroIsUndefined = BuiltinOp != Builtin::BI__lzcnt16 &&
                         BuiltinOp != Builtin::BI__lzcnt &&
                         BuiltinOp != Builtin::BI__lzcnt64;

  if (Val == 0) {
    if (Fallback) {
      pushInteger(S, *Fallback, Call->getType());
      return true;
    }

    if (ZeroIsUndefined)
      return false;
  }

  pushInteger(S, Val.countl_zero(), Call->getType());
  return true;
}

static bool interp__builtin_ctz(InterpState &S, CodePtr OpPC,
                                const InterpFrame *Frame, const CallExpr *Call,
                                unsigned BuiltinID) {
  std::optional<APSInt> Fallback;
  if (BuiltinID == Builtin::BI__builtin_ctzg && Call->getNumArgs() == 2) {
    PrimType FallbackT = *S.getContext().classify(Call->getArg(1));
    Fallback = popToAPSInt(S.Stk, FallbackT);
  }
  APSInt Val;
  if (Call->getArg(0)->getType()->isExtVectorBoolType()) {
    const Pointer &Arg = S.Stk.pop<Pointer>();
    Val = convertBoolVectorToInt(Arg);
  } else {
    PrimType ValT = *S.getContext().classify(Call->getArg(0));
    Val = popToAPSInt(S.Stk, ValT);
  }

  if (Val == 0) {
    if (Fallback) {
      pushInteger(S, *Fallback, Call->getType());
      return true;
    }
    return false;
  }

  pushInteger(S, Val.countr_zero(), Call->getType());
  return true;
}

static bool interp__builtin_bswap(InterpState &S, CodePtr OpPC,
                                  const InterpFrame *Frame,
                                  const CallExpr *Call) {
  PrimType ReturnT = *S.getContext().classify(Call->getType());
  PrimType ValT = *S.getContext().classify(Call->getArg(0));
  const APSInt &Val = popToAPSInt(S.Stk, ValT);
  assert(Val.getActiveBits() <= 64);

  INT_TYPE_SWITCH(ReturnT,
                  { S.Stk.push<T>(T::from(Val.byteSwap().getZExtValue())); });
  return true;
}

/// bool __atomic_always_lock_free(size_t, void const volatile*)
/// bool __atomic_is_lock_free(size_t, void const volatile*)
static bool interp__builtin_atomic_lock_free(InterpState &S, CodePtr OpPC,
                                             const InterpFrame *Frame,
                                             const CallExpr *Call,
                                             unsigned BuiltinOp) {
  auto returnBool = [&S](bool Value) -> bool {
    S.Stk.push<Boolean>(Value);
    return true;
  };

  PrimType ValT = *S.getContext().classify(Call->getArg(0));
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  const APSInt &SizeVal = popToAPSInt(S.Stk, ValT);

  // For __atomic_is_lock_free(sizeof(_Atomic(T))), if the size is a power
  // of two less than or equal to the maximum inline atomic width, we know it
  // is lock-free.  If the size isn't a power of two, or greater than the
  // maximum alignment where we promote atomics, we know it is not lock-free
  // (at least not in the sense of atomic_is_lock_free).  Otherwise,
  // the answer can only be determined at runtime; for example, 16-byte
  // atomics have lock-free implementations on some, but not all,
  // x86-64 processors.

  // Check power-of-two.
  CharUnits Size = CharUnits::fromQuantity(SizeVal.getZExtValue());
  if (Size.isPowerOfTwo()) {
    // Check against inlining width.
    unsigned InlineWidthBits =
        S.getASTContext().getTargetInfo().getMaxAtomicInlineWidth();
    if (Size <= S.getASTContext().toCharUnitsFromBits(InlineWidthBits)) {

      // OK, we will inline appropriately-aligned operations of this size,
      // and _Atomic(T) is appropriately-aligned.
      if (Size == CharUnits::One())
        return returnBool(true);

      // Same for null pointers.
      assert(BuiltinOp != Builtin::BI__c11_atomic_is_lock_free);
      if (Ptr.isZero())
        return returnBool(true);

      if (Ptr.isIntegralPointer()) {
        uint64_t IntVal = Ptr.getIntegerRepresentation();
        if (APSInt(APInt(64, IntVal, false), true).isAligned(Size.getAsAlign()))
          return returnBool(true);
      }

      const Expr *PtrArg = Call->getArg(1);
      // Otherwise, check if the type's alignment against Size.
      if (const auto *ICE = dyn_cast<ImplicitCastExpr>(PtrArg)) {
        // Drop the potential implicit-cast to 'const volatile void*', getting
        // the underlying type.
        if (ICE->getCastKind() == CK_BitCast)
          PtrArg = ICE->getSubExpr();
      }

      if (const auto *PtrTy = PtrArg->getType()->getAs<PointerType>()) {
        QualType PointeeType = PtrTy->getPointeeType();
        if (!PointeeType->isIncompleteType() &&
            S.getASTContext().getTypeAlignInChars(PointeeType) >= Size) {
          // OK, we will inline operations on this object.
          return returnBool(true);
        }
      }
    }
  }

  if (BuiltinOp == Builtin::BI__atomic_always_lock_free)
    return returnBool(false);

  return false;
}

/// bool __c11_atomic_is_lock_free(size_t)
static bool interp__builtin_c11_atomic_is_lock_free(InterpState &S,
                                                    CodePtr OpPC,
                                                    const InterpFrame *Frame,
                                                    const CallExpr *Call) {
  PrimType ValT = *S.getContext().classify(Call->getArg(0));
  const APSInt &SizeVal = popToAPSInt(S.Stk, ValT);

  auto returnBool = [&S](bool Value) -> bool {
    S.Stk.push<Boolean>(Value);
    return true;
  };

  CharUnits Size = CharUnits::fromQuantity(SizeVal.getZExtValue());
  if (Size.isPowerOfTwo()) {
    // Check against inlining width.
    unsigned InlineWidthBits =
        S.getASTContext().getTargetInfo().getMaxAtomicInlineWidth();
    if (Size <= S.getASTContext().toCharUnitsFromBits(InlineWidthBits))
      return returnBool(true);
  }

  return false; // returnBool(false);
}

/// __builtin_complex(Float A, float B);
static bool interp__builtin_complex(InterpState &S, CodePtr OpPC,
                                    const InterpFrame *Frame,
                                    const CallExpr *Call) {
  const Floating &Arg2 = S.Stk.pop<Floating>();
  const Floating &Arg1 = S.Stk.pop<Floating>();
  Pointer &Result = S.Stk.peek<Pointer>();

  Result.elem<Floating>(0) = Arg1;
  Result.elem<Floating>(1) = Arg2;
  Result.initializeAllElements();

  return true;
}

/// __builtin_is_aligned()
/// __builtin_align_up()
/// __builtin_align_down()
/// The first parameter is either an integer or a pointer.
/// The second parameter is the requested alignment as an integer.
static bool interp__builtin_is_aligned_up_down(InterpState &S, CodePtr OpPC,
                                               const InterpFrame *Frame,
                                               const CallExpr *Call,
                                               unsigned BuiltinOp) {
  PrimType AlignmentT = *S.Ctx.classify(Call->getArg(1));
  const APSInt &Alignment = popToAPSInt(S.Stk, AlignmentT);

  if (Alignment < 0 || !Alignment.isPowerOf2()) {
    S.FFDiag(Call, diag::note_constexpr_invalid_alignment) << Alignment;
    return false;
  }
  unsigned SrcWidth = S.getASTContext().getIntWidth(Call->getArg(0)->getType());
  APSInt MaxValue(APInt::getOneBitSet(SrcWidth, SrcWidth - 1));
  if (APSInt::compareValues(Alignment, MaxValue) > 0) {
    S.FFDiag(Call, diag::note_constexpr_alignment_too_big)
        << MaxValue << Call->getArg(0)->getType() << Alignment;
    return false;
  }

  // The first parameter is either an integer or a pointer (but not a function
  // pointer).
  PrimType FirstArgT = *S.Ctx.classify(Call->getArg(0));

  if (isIntegralType(FirstArgT)) {
    const APSInt &Src = popToAPSInt(S.Stk, FirstArgT);
    APInt AlignMinusOne = Alignment.extOrTrunc(Src.getBitWidth()) - 1;
    if (BuiltinOp == Builtin::BI__builtin_align_up) {
      APSInt AlignedVal =
          APSInt((Src + AlignMinusOne) & ~AlignMinusOne, Src.isUnsigned());
      pushInteger(S, AlignedVal, Call->getType());
    } else if (BuiltinOp == Builtin::BI__builtin_align_down) {
      APSInt AlignedVal = APSInt(Src & ~AlignMinusOne, Src.isUnsigned());
      pushInteger(S, AlignedVal, Call->getType());
    } else {
      assert(*S.Ctx.classify(Call->getType()) == PT_Bool);
      S.Stk.push<Boolean>((Src & AlignMinusOne) == 0);
    }
    return true;
  }

  assert(FirstArgT == PT_Ptr);
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  unsigned PtrOffset = Ptr.getByteOffset();
  PtrOffset = Ptr.getIndex();
  CharUnits BaseAlignment =
      S.getASTContext().getDeclAlign(Ptr.getDeclDesc()->asValueDecl());
  CharUnits PtrAlign =
      BaseAlignment.alignmentAtOffset(CharUnits::fromQuantity(PtrOffset));

  if (BuiltinOp == Builtin::BI__builtin_is_aligned) {
    if (PtrAlign.getQuantity() >= Alignment) {
      S.Stk.push<Boolean>(true);
      return true;
    }
    // If the alignment is not known to be sufficient, some cases could still
    // be aligned at run time. However, if the requested alignment is less or
    // equal to the base alignment and the offset is not aligned, we know that
    // the run-time value can never be aligned.
    if (BaseAlignment.getQuantity() >= Alignment &&
        PtrAlign.getQuantity() < Alignment) {
      S.Stk.push<Boolean>(false);
      return true;
    }

    S.FFDiag(Call->getArg(0), diag::note_constexpr_alignment_compute)
        << Alignment;
    return false;
  }

  assert(BuiltinOp == Builtin::BI__builtin_align_down ||
         BuiltinOp == Builtin::BI__builtin_align_up);

  // For align_up/align_down, we can return the same value if the alignment
  // is known to be greater or equal to the requested value.
  if (PtrAlign.getQuantity() >= Alignment) {
    S.Stk.push<Pointer>(Ptr);
    return true;
  }

  // The alignment could be greater than the minimum at run-time, so we cannot
  // infer much about the resulting pointer value. One case is possible:
  // For `_Alignas(32) char buf[N]; __builtin_align_down(&buf[idx], 32)` we
  // can infer the correct index if the requested alignment is smaller than
  // the base alignment so we can perform the computation on the offset.
  if (BaseAlignment.getQuantity() >= Alignment) {
    assert(Alignment.getBitWidth() <= 64 &&
           "Cannot handle > 64-bit address-space");
    uint64_t Alignment64 = Alignment.getZExtValue();
    CharUnits NewOffset =
        CharUnits::fromQuantity(BuiltinOp == Builtin::BI__builtin_align_down
                                    ? llvm::alignDown(PtrOffset, Alignment64)
                                    : llvm::alignTo(PtrOffset, Alignment64));

    S.Stk.push<Pointer>(Ptr.atIndex(NewOffset.getQuantity()));
    return true;
  }

  // Otherwise, we cannot constant-evaluate the result.
  S.FFDiag(Call->getArg(0), diag::note_constexpr_alignment_adjust) << Alignment;
  return false;
}

/// __builtin_assume_aligned(Ptr, Alignment[, ExtraOffset])
static bool interp__builtin_assume_aligned(InterpState &S, CodePtr OpPC,
                                           const InterpFrame *Frame,
                                           const CallExpr *Call) {
  assert(Call->getNumArgs() == 2 || Call->getNumArgs() == 3);

  std::optional<APSInt> ExtraOffset;
  if (Call->getNumArgs() == 3)
    ExtraOffset = popToAPSInt(S.Stk, *S.Ctx.classify(Call->getArg(2)));

  APSInt Alignment = popToAPSInt(S.Stk, *S.Ctx.classify(Call->getArg(1)));
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  CharUnits Align = CharUnits::fromQuantity(Alignment.getZExtValue());

  // If there is a base object, then it must have the correct alignment.
  if (Ptr.isBlockPointer()) {
    CharUnits BaseAlignment;
    if (const auto *VD = Ptr.getDeclDesc()->asValueDecl())
      BaseAlignment = S.getASTContext().getDeclAlign(VD);
    else if (const auto *E = Ptr.getDeclDesc()->asExpr())
      BaseAlignment = GetAlignOfExpr(S.getASTContext(), E, UETT_AlignOf);

    if (BaseAlignment < Align) {
      S.CCEDiag(Call->getArg(0),
                diag::note_constexpr_baa_insufficient_alignment)
          << 0 << BaseAlignment.getQuantity() << Align.getQuantity();
      return false;
    }
  }

  APValue AV = Ptr.toAPValue(S.getASTContext());
  CharUnits AVOffset = AV.getLValueOffset();
  if (ExtraOffset)
    AVOffset -= CharUnits::fromQuantity(ExtraOffset->getZExtValue());
  if (AVOffset.alignTo(Align) != AVOffset) {
    if (Ptr.isBlockPointer())
      S.CCEDiag(Call->getArg(0),
                diag::note_constexpr_baa_insufficient_alignment)
          << 1 << AVOffset.getQuantity() << Align.getQuantity();
    else
      S.CCEDiag(Call->getArg(0),
                diag::note_constexpr_baa_value_insufficient_alignment)
          << AVOffset.getQuantity() << Align.getQuantity();
    return false;
  }

  S.Stk.push<Pointer>(Ptr);
  return true;
}

static bool interp__builtin_ia32_bextr(InterpState &S, CodePtr OpPC,
                                       const InterpFrame *Frame,
                                       const CallExpr *Call) {
  if (Call->getNumArgs() != 2 || !Call->getArg(0)->getType()->isIntegerType() ||
      !Call->getArg(1)->getType()->isIntegerType())
    return false;

  PrimType ValT = *S.Ctx.classify(Call->getArg(0));
  PrimType IndexT = *S.Ctx.classify(Call->getArg(1));
  APSInt Index = popToAPSInt(S.Stk, IndexT);
  APSInt Val = popToAPSInt(S.Stk, ValT);

  unsigned BitWidth = Val.getBitWidth();
  uint64_t Shift = Index.extractBitsAsZExtValue(8, 0);
  uint64_t Length = Index.extractBitsAsZExtValue(8, 8);
  Length = Length > BitWidth ? BitWidth : Length;

  // Handle out of bounds cases.
  if (Length == 0 || Shift >= BitWidth) {
    pushInteger(S, 0, Call->getType());
    return true;
  }

  uint64_t Result = Val.getZExtValue() >> Shift;
  Result &= llvm::maskTrailingOnes<uint64_t>(Length);
  pushInteger(S, Result, Call->getType());
  return true;
}

static bool interp__builtin_ia32_bzhi(InterpState &S, CodePtr OpPC,
                                      const InterpFrame *Frame,
                                      const CallExpr *Call) {
  QualType CallType = Call->getType();
  if (Call->getNumArgs() != 2 || !Call->getArg(0)->getType()->isIntegerType() ||
      !Call->getArg(1)->getType()->isIntegerType() ||
      !CallType->isIntegerType())
    return false;

  PrimType ValT = *S.Ctx.classify(Call->getArg(0));
  PrimType IndexT = *S.Ctx.classify(Call->getArg(1));

  APSInt Idx = popToAPSInt(S.Stk, IndexT);
  APSInt Val = popToAPSInt(S.Stk, ValT);

  unsigned BitWidth = Val.getBitWidth();
  uint64_t Index = Idx.extractBitsAsZExtValue(8, 0);

  if (Index < BitWidth)
    Val.clearHighBits(BitWidth - Index);

  pushInteger(S, Val, CallType);
  return true;
}

static bool interp__builtin_ia32_lzcnt(InterpState &S, CodePtr OpPC,
                                       const InterpFrame *Frame,
                                       const CallExpr *Call) {
  QualType CallType = Call->getType();
  if (!CallType->isIntegerType() ||
      !Call->getArg(0)->getType()->isIntegerType())
    return false;

  APSInt Val = popToAPSInt(S.Stk, *S.Ctx.classify(Call->getArg(0)));
  pushInteger(S, Val.countLeadingZeros(), CallType);
  return true;
}

static bool interp__builtin_ia32_tzcnt(InterpState &S, CodePtr OpPC,
                                       const InterpFrame *Frame,
                                       const CallExpr *Call) {
  QualType CallType = Call->getType();
  if (!CallType->isIntegerType() ||
      !Call->getArg(0)->getType()->isIntegerType())
    return false;

  APSInt Val = popToAPSInt(S.Stk, *S.Ctx.classify(Call->getArg(0)));
  pushInteger(S, Val.countTrailingZeros(), CallType);
  return true;
}

static bool interp__builtin_ia32_pdep(InterpState &S, CodePtr OpPC,
                                      const InterpFrame *Frame,
                                      const CallExpr *Call) {
  if (Call->getNumArgs() != 2 || !Call->getArg(0)->getType()->isIntegerType() ||
      !Call->getArg(1)->getType()->isIntegerType())
    return false;

  PrimType ValT = *S.Ctx.classify(Call->getArg(0));
  PrimType MaskT = *S.Ctx.classify(Call->getArg(1));

  APSInt Mask = popToAPSInt(S.Stk, MaskT);
  APSInt Val = popToAPSInt(S.Stk, ValT);

  unsigned BitWidth = Val.getBitWidth();
  APInt Result = APInt::getZero(BitWidth);
  for (unsigned I = 0, P = 0; I != BitWidth; ++I) {
    if (Mask[I])
      Result.setBitVal(I, Val[P++]);
  }
  pushInteger(S, std::move(Result), Call->getType());
  return true;
}

static bool interp__builtin_ia32_pext(InterpState &S, CodePtr OpPC,
                                      const InterpFrame *Frame,
                                      const CallExpr *Call) {
  if (Call->getNumArgs() != 2 || !Call->getArg(0)->getType()->isIntegerType() ||
      !Call->getArg(1)->getType()->isIntegerType())
    return false;

  PrimType ValT = *S.Ctx.classify(Call->getArg(0));
  PrimType MaskT = *S.Ctx.classify(Call->getArg(1));

  APSInt Mask = popToAPSInt(S.Stk, MaskT);
  APSInt Val = popToAPSInt(S.Stk, ValT);

  unsigned BitWidth = Val.getBitWidth();
  APInt Result = APInt::getZero(BitWidth);
  for (unsigned I = 0, P = 0; I != BitWidth; ++I) {
    if (Mask[I])
      Result.setBitVal(P++, Val[I]);
  }
  pushInteger(S, std::move(Result), Call->getType());
  return true;
}

/// (CarryIn, LHS, RHS, Result)
static bool interp__builtin_ia32_addcarry_subborrow(InterpState &S,
                                                    CodePtr OpPC,
                                                    const InterpFrame *Frame,
                                                    const CallExpr *Call,
                                                    unsigned BuiltinOp) {
  if (Call->getNumArgs() != 4 || !Call->getArg(0)->getType()->isIntegerType() ||
      !Call->getArg(1)->getType()->isIntegerType() ||
      !Call->getArg(2)->getType()->isIntegerType())
    return false;

  const Pointer &CarryOutPtr = S.Stk.pop<Pointer>();

  PrimType CarryInT = *S.getContext().classify(Call->getArg(0));
  PrimType LHST = *S.getContext().classify(Call->getArg(1));
  PrimType RHST = *S.getContext().classify(Call->getArg(2));
  APSInt RHS = popToAPSInt(S.Stk, RHST);
  APSInt LHS = popToAPSInt(S.Stk, LHST);
  APSInt CarryIn = popToAPSInt(S.Stk, CarryInT);

  bool IsAdd = BuiltinOp == clang::X86::BI__builtin_ia32_addcarryx_u32 ||
               BuiltinOp == clang::X86::BI__builtin_ia32_addcarryx_u64;

  unsigned BitWidth = LHS.getBitWidth();
  unsigned CarryInBit = CarryIn.ugt(0) ? 1 : 0;
  APInt ExResult =
      IsAdd ? (LHS.zext(BitWidth + 1) + (RHS.zext(BitWidth + 1) + CarryInBit))
            : (LHS.zext(BitWidth + 1) - (RHS.zext(BitWidth + 1) + CarryInBit));

  APInt Result = ExResult.extractBits(BitWidth, 0);
  APSInt CarryOut =
      APSInt(ExResult.extractBits(1, BitWidth), /*IsUnsigned=*/true);

  QualType CarryOutType = Call->getArg(3)->getType()->getPointeeType();
  PrimType CarryOutT = *S.getContext().classify(CarryOutType);
  assignInteger(S, CarryOutPtr, CarryOutT, APSInt(std::move(Result), true));

  pushInteger(S, CarryOut, Call->getType());

  return true;
}

static bool interp__builtin_os_log_format_buffer_size(InterpState &S,
                                                      CodePtr OpPC,
                                                      const InterpFrame *Frame,
                                                      const CallExpr *Call) {
  analyze_os_log::OSLogBufferLayout Layout;
  analyze_os_log::computeOSLogBufferLayout(S.getASTContext(), Call, Layout);
  pushInteger(S, Layout.size().getQuantity(), Call->getType());
  return true;
}

static bool
interp__builtin_ptrauth_string_discriminator(InterpState &S, CodePtr OpPC,
                                             const InterpFrame *Frame,
                                             const CallExpr *Call) {
  const auto &Ptr = S.Stk.pop<Pointer>();
  assert(Ptr.getFieldDesc()->isPrimitiveArray());

  // This should be created for a StringLiteral, so should alway shold at least
  // one array element.
  assert(Ptr.getFieldDesc()->getNumElems() >= 1);
  StringRef R(&Ptr.deref<char>(), Ptr.getFieldDesc()->getNumElems() - 1);
  uint64_t Result = getPointerAuthStableSipHash(R);
  pushInteger(S, Result, Call->getType());
  return true;
}

static bool interp__builtin_operator_new(InterpState &S, CodePtr OpPC,
                                         const InterpFrame *Frame,
                                         const CallExpr *Call) {
  // A call to __operator_new is only valid within std::allocate<>::allocate.
  // Walk up the call stack to find the appropriate caller and get the
  // element type from it.
  auto [NewCall, ElemType] = S.getStdAllocatorCaller("allocate");

  if (ElemType.isNull()) {
    S.FFDiag(Call, S.getLangOpts().CPlusPlus20
                       ? diag::note_constexpr_new_untyped
                       : diag::note_constexpr_new);
    return false;
  }
  assert(NewCall);

  if (ElemType->isIncompleteType() || ElemType->isFunctionType()) {
    S.FFDiag(Call, diag::note_constexpr_new_not_complete_object_type)
        << (ElemType->isIncompleteType() ? 0 : 1) << ElemType;
    return false;
  }

  // We only care about the first parameter (the size), so discard all the
  // others.
  {
    unsigned NumArgs = Call->getNumArgs();
    assert(NumArgs >= 1);

    // The std::nothrow_t arg never gets put on the stack.
    if (Call->getArg(NumArgs - 1)->getType()->isNothrowT())
      --NumArgs;
    auto Args = ArrayRef(Call->getArgs(), Call->getNumArgs());
    // First arg is needed.
    Args = Args.drop_front();

    // Discard the rest.
    for (const Expr *Arg : Args)
      discard(S.Stk, *S.getContext().classify(Arg));
  }

  APSInt Bytes = popToAPSInt(S.Stk, *S.getContext().classify(Call->getArg(0)));
  CharUnits ElemSize = S.getASTContext().getTypeSizeInChars(ElemType);
  assert(!ElemSize.isZero());
  // Divide the number of bytes by sizeof(ElemType), so we get the number of
  // elements we should allocate.
  APInt NumElems, Remainder;
  APInt ElemSizeAP(Bytes.getBitWidth(), ElemSize.getQuantity());
  APInt::udivrem(Bytes, ElemSizeAP, NumElems, Remainder);
  if (Remainder != 0) {
    // This likely indicates a bug in the implementation of 'std::allocator'.
    S.FFDiag(Call, diag::note_constexpr_operator_new_bad_size)
        << Bytes << APSInt(ElemSizeAP, true) << ElemType;
    return false;
  }

  // NB: The same check we're using in CheckArraySize()
  if (NumElems.getActiveBits() >
          ConstantArrayType::getMaxSizeBits(S.getASTContext()) ||
      NumElems.ugt(Descriptor::MaxArrayElemBytes / ElemSize.getQuantity())) {
    // FIXME: NoThrow check?
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_constexpr_new_too_large)
        << NumElems.getZExtValue();
    return false;
  }

  if (!CheckArraySize(S, OpPC, NumElems.getZExtValue()))
    return false;

  bool IsArray = NumElems.ugt(1);
  OptPrimType ElemT = S.getContext().classify(ElemType);
  DynamicAllocator &Allocator = S.getAllocator();
  if (ElemT) {
    Block *B =
        Allocator.allocate(NewCall, *ElemT, NumElems.getZExtValue(),
                           S.Ctx.getEvalID(), DynamicAllocator::Form::Operator);
    assert(B);
    S.Stk.push<Pointer>(Pointer(B).atIndex(0));
    return true;
  }

  assert(!ElemT);

  // Composite arrays
  if (IsArray) {
    const Descriptor *Desc =
        S.P.createDescriptor(NewCall, ElemType.getTypePtr(), std::nullopt);
    Block *B =
        Allocator.allocate(Desc, NumElems.getZExtValue(), S.Ctx.getEvalID(),
                           DynamicAllocator::Form::Operator);
    assert(B);
    S.Stk.push<Pointer>(Pointer(B).atIndex(0));
    return true;
  }

  // Records. Still allocate them as single-element arrays.
  QualType AllocType = S.getASTContext().getConstantArrayType(
      ElemType, NumElems, nullptr, ArraySizeModifier::Normal, 0);

  const Descriptor *Desc = S.P.createDescriptor(NewCall, AllocType.getTypePtr(),
                                                Descriptor::InlineDescMD);
  Block *B = Allocator.allocate(Desc, S.getContext().getEvalID(),
                                DynamicAllocator::Form::Operator);
  assert(B);
  S.Stk.push<Pointer>(Pointer(B).atIndex(0).narrow());
  return true;
}

static bool interp__builtin_operator_delete(InterpState &S, CodePtr OpPC,
                                            const InterpFrame *Frame,
                                            const CallExpr *Call) {
  const Expr *Source = nullptr;
  const Block *BlockToDelete = nullptr;

  if (S.checkingPotentialConstantExpression()) {
    S.Stk.discard<Pointer>();
    return false;
  }

  // This is permitted only within a call to std::allocator<T>::deallocate.
  if (!S.getStdAllocatorCaller("deallocate")) {
    S.FFDiag(Call);
    S.Stk.discard<Pointer>();
    return true;
  }

  {
    const Pointer &Ptr = S.Stk.pop<Pointer>();

    if (Ptr.isZero()) {
      S.CCEDiag(Call, diag::note_constexpr_deallocate_null);
      return true;
    }

    Source = Ptr.getDeclDesc()->asExpr();
    BlockToDelete = Ptr.block();

    if (!BlockToDelete->isDynamic()) {
      S.FFDiag(Call, diag::note_constexpr_delete_not_heap_alloc)
          << Ptr.toDiagnosticString(S.getASTContext());
      if (const auto *D = Ptr.getFieldDesc()->asDecl())
        S.Note(D->getLocation(), diag::note_declared_at);
    }
  }
  assert(BlockToDelete);

  DynamicAllocator &Allocator = S.getAllocator();
  const Descriptor *BlockDesc = BlockToDelete->getDescriptor();
  std::optional<DynamicAllocator::Form> AllocForm =
      Allocator.getAllocationForm(Source);

  if (!Allocator.deallocate(Source, BlockToDelete, S)) {
    // Nothing has been deallocated, this must be a double-delete.
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_constexpr_double_delete);
    return false;
  }
  assert(AllocForm);

  return CheckNewDeleteForms(
      S, OpPC, *AllocForm, DynamicAllocator::Form::Operator, BlockDesc, Source);
}

static bool interp__builtin_arithmetic_fence(InterpState &S, CodePtr OpPC,
                                             const InterpFrame *Frame,
                                             const CallExpr *Call) {
  const Floating &Arg0 = S.Stk.pop<Floating>();
  S.Stk.push<Floating>(Arg0);
  return true;
}

static bool interp__builtin_vector_reduce(InterpState &S, CodePtr OpPC,
                                          const CallExpr *Call, unsigned ID) {
  const Pointer &Arg = S.Stk.pop<Pointer>();
  assert(Arg.getFieldDesc()->isPrimitiveArray());

  QualType ElemType = Arg.getFieldDesc()->getElemQualType();
  assert(Call->getType() == ElemType);
  PrimType ElemT = *S.getContext().classify(ElemType);
  unsigned NumElems = Arg.getNumElems();

  INT_TYPE_SWITCH_NO_BOOL(ElemT, {
    T Result = Arg.elem<T>(0);
    unsigned BitWidth = Result.bitWidth();
    for (unsigned I = 1; I != NumElems; ++I) {
      T Elem = Arg.elem<T>(I);
      T PrevResult = Result;

      if (ID == Builtin::BI__builtin_reduce_add) {
        if (T::add(Result, Elem, BitWidth, &Result)) {
          unsigned OverflowBits = BitWidth + 1;
          (void)handleOverflow(S, OpPC,
                               (PrevResult.toAPSInt(OverflowBits) +
                                Elem.toAPSInt(OverflowBits)));
          return false;
        }
      } else if (ID == Builtin::BI__builtin_reduce_mul) {
        if (T::mul(Result, Elem, BitWidth, &Result)) {
          unsigned OverflowBits = BitWidth * 2;
          (void)handleOverflow(S, OpPC,
                               (PrevResult.toAPSInt(OverflowBits) *
                                Elem.toAPSInt(OverflowBits)));
          return false;
        }

      } else if (ID == Builtin::BI__builtin_reduce_and) {
        (void)T::bitAnd(Result, Elem, BitWidth, &Result);
      } else if (ID == Builtin::BI__builtin_reduce_or) {
        (void)T::bitOr(Result, Elem, BitWidth, &Result);
      } else if (ID == Builtin::BI__builtin_reduce_xor) {
        (void)T::bitXor(Result, Elem, BitWidth, &Result);
      } else if (ID == Builtin::BI__builtin_reduce_min) {
        if (Elem < Result)
          Result = Elem;
      } else if (ID == Builtin::BI__builtin_reduce_max) {
        if (Elem > Result)
          Result = Elem;
      } else {
        llvm_unreachable("Unhandled vector reduce builtin");
      }
    }
    pushInteger(S, Result.toAPSInt(), Call->getType());
  });

  return true;
}

static bool interp__builtin_elementwise_abs(InterpState &S, CodePtr OpPC,
                                            const InterpFrame *Frame,
                                            const CallExpr *Call,
                                            unsigned BuiltinID) {
  assert(Call->getNumArgs() == 1);
  QualType Ty = Call->getArg(0)->getType();
  if (Ty->isIntegerType()) {
    PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
    APSInt Val = popToAPSInt(S.Stk, ArgT);

    pushInteger(S, Val.abs(), Call->getType());
    return true;
  }

  if (Ty->isFloatingType()) {
    Floating Val = S.Stk.pop<Floating>();
    Floating Result = abs(S, Val);
    S.Stk.push<Floating>(Result);
    return true;
  }

  // Otherwise, the argument must be a vector.
  assert(Call->getArg(0)->getType()->isVectorType());
  const Pointer &Arg = S.Stk.pop<Pointer>();
  assert(Arg.getFieldDesc()->isPrimitiveArray());
  const Pointer &Dst = S.Stk.peek<Pointer>();
  assert(Dst.getFieldDesc()->isPrimitiveArray());
  assert(Arg.getFieldDesc()->getNumElems() ==
         Dst.getFieldDesc()->getNumElems());

  QualType ElemType = Arg.getFieldDesc()->getElemQualType();
  PrimType ElemT = *S.getContext().classify(ElemType);
  unsigned NumElems = Arg.getNumElems();
  // we can either have a vector of integer or a vector of floating point
  for (unsigned I = 0; I != NumElems; ++I) {
    if (ElemType->isIntegerType()) {
      INT_TYPE_SWITCH_NO_BOOL(ElemT, {
        Dst.elem<T>(I) = T::from(static_cast<T>(
            APSInt(Arg.elem<T>(I).toAPSInt().abs(),
                   ElemType->isUnsignedIntegerOrEnumerationType())));
      });
    } else {
      Floating Val = Arg.elem<Floating>(I);
      Dst.elem<Floating>(I) = abs(S, Val);
    }
  }
  Dst.initializeAllElements();

  return true;
}

/// Can be called with an integer or vector as the first and only parameter.
static bool interp__builtin_elementwise_popcount(InterpState &S, CodePtr OpPC,
                                                 const InterpFrame *Frame,
                                                 const CallExpr *Call,
                                                 unsigned BuiltinID) {
  assert(Call->getNumArgs() == 1);
  if (Call->getArg(0)->getType()->isIntegerType()) {
    PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
    APSInt Val = popToAPSInt(S.Stk, ArgT);

    if (BuiltinID == Builtin::BI__builtin_elementwise_popcount) {
      pushInteger(S, Val.popcount(), Call->getType());
    } else {
      pushInteger(S, Val.reverseBits(), Call->getType());
    }
    return true;
  }
  // Otherwise, the argument must be a vector.
  assert(Call->getArg(0)->getType()->isVectorType());
  const Pointer &Arg = S.Stk.pop<Pointer>();
  assert(Arg.getFieldDesc()->isPrimitiveArray());
  const Pointer &Dst = S.Stk.peek<Pointer>();
  assert(Dst.getFieldDesc()->isPrimitiveArray());
  assert(Arg.getFieldDesc()->getNumElems() ==
         Dst.getFieldDesc()->getNumElems());

  QualType ElemType = Arg.getFieldDesc()->getElemQualType();
  PrimType ElemT = *S.getContext().classify(ElemType);
  unsigned NumElems = Arg.getNumElems();

  // FIXME: Reading from uninitialized vector elements?
  for (unsigned I = 0; I != NumElems; ++I) {
    INT_TYPE_SWITCH_NO_BOOL(ElemT, {
      if (BuiltinID == Builtin::BI__builtin_elementwise_popcount) {
        Dst.elem<T>(I) = T::from(Arg.elem<T>(I).toAPSInt().popcount());
      } else {
        Dst.elem<T>(I) =
            T::from(Arg.elem<T>(I).toAPSInt().reverseBits().getZExtValue());
      }
    });
  }
  Dst.initializeAllElements();

  return true;
}

/// Can be called with an integer or vector as the first and only parameter.
static bool interp__builtin_elementwise_countzeroes(InterpState &S,
                                                    CodePtr OpPC,
                                                    const InterpFrame *Frame,
                                                    const CallExpr *Call,
                                                    unsigned BuiltinID) {
  const bool HasZeroArg = Call->getNumArgs() == 2;
  const bool IsCTTZ = BuiltinID == Builtin::BI__builtin_elementwise_cttz;
  assert(Call->getNumArgs() == 1 || HasZeroArg);
  if (Call->getArg(0)->getType()->isIntegerType()) {
    PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
    APSInt Val = popToAPSInt(S.Stk, ArgT);
    std::optional<APSInt> ZeroVal;
    if (HasZeroArg) {
      ZeroVal = Val;
      Val = popToAPSInt(S.Stk, ArgT);
    }

    if (Val.isZero()) {
      if (ZeroVal) {
        pushInteger(S, *ZeroVal, Call->getType());
        return true;
      }
      // If we haven't been provided the second argument, the result is
      // undefined
      S.FFDiag(S.Current->getSource(OpPC),
               diag::note_constexpr_countzeroes_zero)
          << /*IsTrailing=*/IsCTTZ;
      return false;
    }

    if (BuiltinID == Builtin::BI__builtin_elementwise_ctlz) {
      pushInteger(S, Val.countLeadingZeros(), Call->getType());
    } else {
      pushInteger(S, Val.countTrailingZeros(), Call->getType());
    }
    return true;
  }
  // Otherwise, the argument must be a vector.
  const ASTContext &ASTCtx = S.getASTContext();
  Pointer ZeroArg;
  if (HasZeroArg) {
    assert(Call->getArg(1)->getType()->isVectorType() &&
           ASTCtx.hasSameUnqualifiedType(Call->getArg(0)->getType(),
                                         Call->getArg(1)->getType()));
    (void)ASTCtx;
    ZeroArg = S.Stk.pop<Pointer>();
    assert(ZeroArg.getFieldDesc()->isPrimitiveArray());
  }
  assert(Call->getArg(0)->getType()->isVectorType());
  const Pointer &Arg = S.Stk.pop<Pointer>();
  assert(Arg.getFieldDesc()->isPrimitiveArray());
  const Pointer &Dst = S.Stk.peek<Pointer>();
  assert(Dst.getFieldDesc()->isPrimitiveArray());
  assert(Arg.getFieldDesc()->getNumElems() ==
         Dst.getFieldDesc()->getNumElems());

  QualType ElemType = Arg.getFieldDesc()->getElemQualType();
  PrimType ElemT = *S.getContext().classify(ElemType);
  unsigned NumElems = Arg.getNumElems();

  // FIXME: Reading from uninitialized vector elements?
  for (unsigned I = 0; I != NumElems; ++I) {
    INT_TYPE_SWITCH_NO_BOOL(ElemT, {
      APInt EltVal = Arg.atIndex(I).deref<T>().toAPSInt();
      if (EltVal.isZero()) {
        if (HasZeroArg) {
          Dst.atIndex(I).deref<T>() = ZeroArg.atIndex(I).deref<T>();
        } else {
          // If we haven't been provided the second argument, the result is
          // undefined
          S.FFDiag(S.Current->getSource(OpPC),
                   diag::note_constexpr_countzeroes_zero)
              << /*IsTrailing=*/IsCTTZ;
          return false;
        }
      } else if (IsCTTZ) {
        Dst.atIndex(I).deref<T>() = T::from(EltVal.countTrailingZeros());
      } else {
        Dst.atIndex(I).deref<T>() = T::from(EltVal.countLeadingZeros());
      }
      Dst.atIndex(I).initialize();
    });
  }

  return true;
}

static bool interp__builtin_memcpy(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const CallExpr *Call, unsigned ID) {
  assert(Call->getNumArgs() == 3);
  const ASTContext &ASTCtx = S.getASTContext();
  PrimType SizeT = *S.getContext().classify(Call->getArg(2));
  APSInt Size = popToAPSInt(S.Stk, SizeT);
  const Pointer SrcPtr = S.Stk.pop<Pointer>();
  const Pointer DestPtr = S.Stk.pop<Pointer>();

  assert(!Size.isSigned() && "memcpy and friends take an unsigned size");

  if (ID == Builtin::BImemcpy || ID == Builtin::BImemmove)
    diagnoseNonConstexprBuiltin(S, OpPC, ID);

  bool Move =
      (ID == Builtin::BI__builtin_memmove || ID == Builtin::BImemmove ||
       ID == Builtin::BI__builtin_wmemmove || ID == Builtin::BIwmemmove);
  bool WChar = ID == Builtin::BIwmemcpy || ID == Builtin::BIwmemmove ||
               ID == Builtin::BI__builtin_wmemcpy ||
               ID == Builtin::BI__builtin_wmemmove;

  // If the size is zero, we treat this as always being a valid no-op.
  if (Size.isZero()) {
    S.Stk.push<Pointer>(DestPtr);
    return true;
  }

  if (SrcPtr.isZero() || DestPtr.isZero()) {
    Pointer DiagPtr = (SrcPtr.isZero() ? SrcPtr : DestPtr);
    S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_memcpy_null)
        << /*IsMove=*/Move << /*IsWchar=*/WChar << !SrcPtr.isZero()
        << DiagPtr.toDiagnosticString(ASTCtx);
    return false;
  }

  // Diagnose integral src/dest pointers specially.
  if (SrcPtr.isIntegralPointer() || DestPtr.isIntegralPointer()) {
    std::string DiagVal = "(void *)";
    DiagVal += SrcPtr.isIntegralPointer()
                   ? std::to_string(SrcPtr.getIntegerRepresentation())
                   : std::to_string(DestPtr.getIntegerRepresentation());
    S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_memcpy_null)
        << Move << WChar << DestPtr.isIntegralPointer() << DiagVal;
    return false;
  }

  // Can't read from dummy pointers.
  if (DestPtr.isDummy() || SrcPtr.isDummy())
    return false;

  if (DestPtr.getType()->isIncompleteType()) {
    S.FFDiag(S.Current->getSource(OpPC),
             diag::note_constexpr_memcpy_incomplete_type)
        << Move << DestPtr.getType();
    return false;
  }
  if (SrcPtr.getType()->isIncompleteType()) {
    S.FFDiag(S.Current->getSource(OpPC),
             diag::note_constexpr_memcpy_incomplete_type)
        << Move << SrcPtr.getType();
    return false;
  }

  QualType DestElemType = getElemType(DestPtr);
  if (DestElemType->isIncompleteType()) {
    S.FFDiag(S.Current->getSource(OpPC),
             diag::note_constexpr_memcpy_incomplete_type)
        << Move << DestElemType;
    return false;
  }

  size_t RemainingDestElems;
  if (DestPtr.getFieldDesc()->isArray()) {
    RemainingDestElems = DestPtr.isUnknownSizeArray()
                             ? 0
                             : (DestPtr.getNumElems() - DestPtr.getIndex());
  } else {
    RemainingDestElems = 1;
  }
  unsigned DestElemSize = ASTCtx.getTypeSizeInChars(DestElemType).getQuantity();

  if (WChar) {
    uint64_t WCharSize =
        ASTCtx.getTypeSizeInChars(ASTCtx.getWCharType()).getQuantity();
    Size *= APSInt(APInt(Size.getBitWidth(), WCharSize, /*IsSigned=*/false),
                   /*IsUnsigend=*/true);
  }

  if (Size.urem(DestElemSize) != 0) {
    S.FFDiag(S.Current->getSource(OpPC),
             diag::note_constexpr_memcpy_unsupported)
        << Move << WChar << 0 << DestElemType << Size << DestElemSize;
    return false;
  }

  QualType SrcElemType = getElemType(SrcPtr);
  size_t RemainingSrcElems;
  if (SrcPtr.getFieldDesc()->isArray()) {
    RemainingSrcElems = SrcPtr.isUnknownSizeArray()
                            ? 0
                            : (SrcPtr.getNumElems() - SrcPtr.getIndex());
  } else {
    RemainingSrcElems = 1;
  }
  unsigned SrcElemSize = ASTCtx.getTypeSizeInChars(SrcElemType).getQuantity();

  if (!ASTCtx.hasSameUnqualifiedType(DestElemType, SrcElemType)) {
    S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_memcpy_type_pun)
        << Move << SrcElemType << DestElemType;
    return false;
  }

  if (!DestElemType.isTriviallyCopyableType(ASTCtx)) {
    S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_memcpy_nontrivial)
        << Move << DestElemType;
    return false;
  }

  // Check if we have enough elements to read from and write to.
  size_t RemainingDestBytes = RemainingDestElems * DestElemSize;
  size_t RemainingSrcBytes = RemainingSrcElems * SrcElemSize;
  if (Size.ugt(RemainingDestBytes) || Size.ugt(RemainingSrcBytes)) {
    APInt N = Size.udiv(DestElemSize);
    S.FFDiag(S.Current->getSource(OpPC),
             diag::note_constexpr_memcpy_unsupported)
        << Move << WChar << (Size.ugt(RemainingSrcBytes) ? 1 : 2)
        << DestElemType << toString(N, 10, /*Signed=*/false);
    return false;
  }

  // Check for overlapping memory regions.
  if (!Move && Pointer::pointToSameBlock(SrcPtr, DestPtr)) {
    // Remove base casts.
    Pointer SrcP = SrcPtr;
    while (SrcP.isBaseClass())
      SrcP = SrcP.getBase();

    Pointer DestP = DestPtr;
    while (DestP.isBaseClass())
      DestP = DestP.getBase();

    unsigned SrcIndex = SrcP.expand().getIndex() * SrcP.elemSize();
    unsigned DstIndex = DestP.expand().getIndex() * DestP.elemSize();
    unsigned N = Size.getZExtValue();

    if ((SrcIndex <= DstIndex && (SrcIndex + N) > DstIndex) ||
        (DstIndex <= SrcIndex && (DstIndex + N) > SrcIndex)) {
      S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_memcpy_overlap)
          << /*IsWChar=*/false;
      return false;
    }
  }

  assert(Size.getZExtValue() % DestElemSize == 0);
  if (!DoMemcpy(S, OpPC, SrcPtr, DestPtr, Bytes(Size.getZExtValue()).toBits()))
    return false;

  S.Stk.push<Pointer>(DestPtr);
  return true;
}

/// Determine if T is a character type for which we guarantee that
/// sizeof(T) == 1.
static bool isOneByteCharacterType(QualType T) {
  return T->isCharType() || T->isChar8Type();
}

static bool interp__builtin_memcmp(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const CallExpr *Call, unsigned ID) {
  assert(Call->getNumArgs() == 3);
  PrimType SizeT = *S.getContext().classify(Call->getArg(2));
  const APSInt &Size = popToAPSInt(S.Stk, SizeT);
  const Pointer &PtrB = S.Stk.pop<Pointer>();
  const Pointer &PtrA = S.Stk.pop<Pointer>();

  if (ID == Builtin::BImemcmp || ID == Builtin::BIbcmp ||
      ID == Builtin::BIwmemcmp)
    diagnoseNonConstexprBuiltin(S, OpPC, ID);

  if (Size.isZero()) {
    pushInteger(S, 0, Call->getType());
    return true;
  }

  bool IsWide =
      (ID == Builtin::BIwmemcmp || ID == Builtin::BI__builtin_wmemcmp);

  const ASTContext &ASTCtx = S.getASTContext();
  QualType ElemTypeA = getElemType(PtrA);
  QualType ElemTypeB = getElemType(PtrB);
  // FIXME: This is an arbitrary limitation the current constant interpreter
  // had. We could remove this.
  if (!IsWide && (!isOneByteCharacterType(ElemTypeA) ||
                  !isOneByteCharacterType(ElemTypeB))) {
    S.FFDiag(S.Current->getSource(OpPC),
             diag::note_constexpr_memcmp_unsupported)
        << ASTCtx.BuiltinInfo.getQuotedName(ID) << PtrA.getType()
        << PtrB.getType();
    return false;
  }

  if (PtrA.isDummy() || PtrB.isDummy())
    return false;

  // Now, read both pointers to a buffer and compare those.
  BitcastBuffer BufferA(
      Bits(ASTCtx.getTypeSize(ElemTypeA) * PtrA.getNumElems()));
  readPointerToBuffer(S.getContext(), PtrA, BufferA, false);
  // FIXME: The swapping here is UNDOING something we do when reading the
  // data into the buffer.
  if (ASTCtx.getTargetInfo().isBigEndian())
    swapBytes(BufferA.Data.get(), BufferA.byteSize().getQuantity());

  BitcastBuffer BufferB(
      Bits(ASTCtx.getTypeSize(ElemTypeB) * PtrB.getNumElems()));
  readPointerToBuffer(S.getContext(), PtrB, BufferB, false);
  // FIXME: The swapping here is UNDOING something we do when reading the
  // data into the buffer.
  if (ASTCtx.getTargetInfo().isBigEndian())
    swapBytes(BufferB.Data.get(), BufferB.byteSize().getQuantity());

  size_t MinBufferSize = std::min(BufferA.byteSize().getQuantity(),
                                  BufferB.byteSize().getQuantity());

  unsigned ElemSize = 1;
  if (IsWide)
    ElemSize = ASTCtx.getTypeSizeInChars(ASTCtx.getWCharType()).getQuantity();
  // The Size given for the wide variants is in wide-char units. Convert it
  // to bytes.
  size_t ByteSize = Size.getZExtValue() * ElemSize;
  size_t CmpSize = std::min(MinBufferSize, ByteSize);

  for (size_t I = 0; I != CmpSize; I += ElemSize) {
    if (IsWide) {
      INT_TYPE_SWITCH(*S.getContext().classify(ASTCtx.getWCharType()), {
        T A = *reinterpret_cast<T *>(BufferA.Data.get() + I);
        T B = *reinterpret_cast<T *>(BufferB.Data.get() + I);
        if (A < B) {
          pushInteger(S, -1, Call->getType());
          return true;
        }
        if (A > B) {
          pushInteger(S, 1, Call->getType());
          return true;
        }
      });
    } else {
      std::byte A = BufferA.Data[I];
      std::byte B = BufferB.Data[I];

      if (A < B) {
        pushInteger(S, -1, Call->getType());
        return true;
      }
      if (A > B) {
        pushInteger(S, 1, Call->getType());
        return true;
      }
    }
  }

  // We compared CmpSize bytes above. If the limiting factor was the Size
  // passed, we're done and the result is equality (0).
  if (ByteSize <= CmpSize) {
    pushInteger(S, 0, Call->getType());
    return true;
  }

  // However, if we read all the available bytes but were instructed to read
  // even more, diagnose this as a "read of dereferenced one-past-the-end
  // pointer". This is what would happen if we called CheckLoad() on every array
  // element.
  S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_access_past_end)
      << AK_Read << S.Current->getRange(OpPC);
  return false;
}

// __builtin_memchr(ptr, int, int)
// __builtin_strchr(ptr, int)
static bool interp__builtin_memchr(InterpState &S, CodePtr OpPC,
                                   const CallExpr *Call, unsigned ID) {
  if (ID == Builtin::BImemchr || ID == Builtin::BIwcschr ||
      ID == Builtin::BIstrchr || ID == Builtin::BIwmemchr)
    diagnoseNonConstexprBuiltin(S, OpPC, ID);

  std::optional<APSInt> MaxLength;
  PrimType DesiredT = *S.getContext().classify(Call->getArg(1));
  if (Call->getNumArgs() == 3) {
    PrimType MaxT = *S.getContext().classify(Call->getArg(2));
    MaxLength = popToAPSInt(S.Stk, MaxT);
  }
  APSInt Desired = popToAPSInt(S.Stk, DesiredT);
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (MaxLength && MaxLength->isZero()) {
    S.Stk.push<Pointer>();
    return true;
  }

  if (Ptr.isDummy()) {
    if (Ptr.getType()->isIncompleteType())
      S.FFDiag(S.Current->getSource(OpPC),
               diag::note_constexpr_ltor_incomplete_type)
          << Ptr.getType();
    return false;
  }

  // Null is only okay if the given size is 0.
  if (Ptr.isZero()) {
    S.FFDiag(S.Current->getSource(OpPC), diag::note_constexpr_access_null)
        << AK_Read;
    return false;
  }

  QualType ElemTy = Ptr.getFieldDesc()->isArray()
                        ? Ptr.getFieldDesc()->getElemQualType()
                        : Ptr.getFieldDesc()->getType();
  bool IsRawByte = ID == Builtin::BImemchr || ID == Builtin::BI__builtin_memchr;

  // Give up on byte-oriented matching against multibyte elements.
  if (IsRawByte && !isOneByteCharacterType(ElemTy)) {
    S.FFDiag(S.Current->getSource(OpPC),
             diag::note_constexpr_memchr_unsupported)
        << S.getASTContext().BuiltinInfo.getQuotedName(ID) << ElemTy;
    return false;
  }

  if (ID == Builtin::BIstrchr || ID == Builtin::BI__builtin_strchr) {
    // strchr compares directly to the passed integer, and therefore
    // always fails if given an int that is not a char.
    if (Desired !=
        Desired.trunc(S.getASTContext().getCharWidth()).getSExtValue()) {
      S.Stk.push<Pointer>();
      return true;
    }
  }

  uint64_t DesiredVal;
  if (ID == Builtin::BIwmemchr || ID == Builtin::BI__builtin_wmemchr ||
      ID == Builtin::BIwcschr || ID == Builtin::BI__builtin_wcschr) {
    // wcschr and wmemchr are given a wchar_t to look for. Just use it.
    DesiredVal = Desired.getZExtValue();
  } else {
    DesiredVal = Desired.trunc(S.getASTContext().getCharWidth()).getZExtValue();
  }

  bool StopAtZero =
      (ID == Builtin::BIstrchr || ID == Builtin::BI__builtin_strchr ||
       ID == Builtin::BIwcschr || ID == Builtin::BI__builtin_wcschr);

  PrimType ElemT =
      IsRawByte ? PT_Sint8 : *S.getContext().classify(getElemType(Ptr));

  size_t Index = Ptr.getIndex();
  size_t Step = 0;
  for (;;) {
    const Pointer &ElemPtr =
        (Index + Step) > 0 ? Ptr.atIndex(Index + Step) : Ptr;

    if (!CheckLoad(S, OpPC, ElemPtr))
      return false;

    uint64_t V;
    INT_TYPE_SWITCH_NO_BOOL(
        ElemT, { V = static_cast<uint64_t>(ElemPtr.deref<T>().toUnsigned()); });

    if (V == DesiredVal) {
      S.Stk.push<Pointer>(ElemPtr);
      return true;
    }

    if (StopAtZero && V == 0)
      break;

    ++Step;
    if (MaxLength && Step == MaxLength->getZExtValue())
      break;
  }

  S.Stk.push<Pointer>();
  return true;
}

static std::optional<unsigned> computeFullDescSize(const ASTContext &ASTCtx,
                                                   const Descriptor *Desc) {
  if (Desc->isPrimitive())
    return ASTCtx.getTypeSizeInChars(Desc->getType()).getQuantity();
  if (Desc->isArray())
    return ASTCtx.getTypeSizeInChars(Desc->getElemQualType()).getQuantity() *
           Desc->getNumElems();
  if (Desc->isRecord()) {
    // Can't use Descriptor::getType() as that may return a pointer type. Look
    // at the decl directly.
    return ASTCtx
        .getTypeSizeInChars(
            ASTCtx.getCanonicalTagType(Desc->ElemRecord->getDecl()))
        .getQuantity();
  }

  return std::nullopt;
}

/// Compute the byte offset of \p Ptr in the full declaration.
static unsigned computePointerOffset(const ASTContext &ASTCtx,
                                     const Pointer &Ptr) {
  unsigned Result = 0;

  Pointer P = Ptr;
  while (P.isField() || P.isArrayElement()) {
    P = P.expand();
    const Descriptor *D = P.getFieldDesc();

    if (P.isArrayElement()) {
      unsigned ElemSize =
          ASTCtx.getTypeSizeInChars(D->getElemQualType()).getQuantity();
      if (P.isOnePastEnd())
        Result += ElemSize * P.getNumElems();
      else
        Result += ElemSize * P.getIndex();
      P = P.expand().getArray();
    } else if (P.isBaseClass()) {
      const auto *RD = cast<CXXRecordDecl>(D->asDecl());
      bool IsVirtual = Ptr.isVirtualBaseClass();
      P = P.getBase();
      const Record *BaseRecord = P.getRecord();

      const ASTRecordLayout &Layout =
          ASTCtx.getASTRecordLayout(cast<CXXRecordDecl>(BaseRecord->getDecl()));
      if (IsVirtual)
        Result += Layout.getVBaseClassOffset(RD).getQuantity();
      else
        Result += Layout.getBaseClassOffset(RD).getQuantity();
    } else if (P.isField()) {
      const FieldDecl *FD = P.getField();
      const ASTRecordLayout &Layout =
          ASTCtx.getASTRecordLayout(FD->getParent());
      unsigned FieldIndex = FD->getFieldIndex();
      uint64_t FieldOffset =
          ASTCtx.toCharUnitsFromBits(Layout.getFieldOffset(FieldIndex))
              .getQuantity();
      Result += FieldOffset;
      P = P.getBase();
    } else
      llvm_unreachable("Unhandled descriptor type");
  }

  return Result;
}

/// Does Ptr point to the last subobject?
static bool pointsToLastObject(const Pointer &Ptr) {
  Pointer P = Ptr;
  while (!P.isRoot()) {

    if (P.isArrayElement()) {
      P = P.expand().getArray();
      continue;
    }
    if (P.isBaseClass()) {
      if (P.getRecord()->getNumFields() > 0)
        return false;
      P = P.getBase();
      continue;
    }

    Pointer Base = P.getBase();
    if (const Record *R = Base.getRecord()) {
      assert(P.getField());
      if (P.getField()->getFieldIndex() != R->getNumFields() - 1)
        return false;
    }
    P = Base;
  }

  return true;
}

/// Does Ptr point to the last object AND to a flexible array member?
static bool isUserWritingOffTheEnd(const ASTContext &Ctx, const Pointer &Ptr) {
  auto isFlexibleArrayMember = [&](const Descriptor *FieldDesc) {
    using FAMKind = LangOptions::StrictFlexArraysLevelKind;
    FAMKind StrictFlexArraysLevel =
        Ctx.getLangOpts().getStrictFlexArraysLevel();

    if (StrictFlexArraysLevel == FAMKind::Default)
      return true;

    unsigned NumElems = FieldDesc->getNumElems();
    if (NumElems == 0 && StrictFlexArraysLevel != FAMKind::IncompleteOnly)
      return true;

    if (NumElems == 1 && StrictFlexArraysLevel == FAMKind::OneZeroOrIncomplete)
      return true;
    return false;
  };

  const Descriptor *FieldDesc = Ptr.getFieldDesc();
  if (!FieldDesc->isArray())
    return false;

  return Ptr.isDummy() && pointsToLastObject(Ptr) &&
         isFlexibleArrayMember(FieldDesc);
}

static bool interp__builtin_object_size(InterpState &S, CodePtr OpPC,
                                        const InterpFrame *Frame,
                                        const CallExpr *Call) {
  const ASTContext &ASTCtx = S.getASTContext();
  PrimType KindT = *S.getContext().classify(Call->getArg(1));
  // From the GCC docs:
  // Kind is an integer constant from 0 to 3. If the least significant bit is
  // clear, objects are whole variables. If it is set, a closest surrounding
  // subobject is considered the object a pointer points to. The second bit
  // determines if maximum or minimum of remaining bytes is computed.
  unsigned Kind = popToAPSInt(S.Stk, KindT).getZExtValue();
  assert(Kind <= 3 && "unexpected kind");
  bool UseFieldDesc = (Kind & 1u);
  bool ReportMinimum = (Kind & 2u);
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (Call->getArg(0)->HasSideEffects(ASTCtx)) {
    // "If there are any side effects in them, it returns (size_t) -1
    // for type 0 or 1 and (size_t) 0 for type 2 or 3."
    pushInteger(S, Kind <= 1 ? -1 : 0, Call->getType());
    return true;
  }

  if (Ptr.isZero() || !Ptr.isBlockPointer())
    return false;

  // We can't load through pointers.
  if (Ptr.isDummy() && Ptr.getType()->isPointerType())
    return false;

  bool DetermineForCompleteObject = Ptr.getFieldDesc() == Ptr.getDeclDesc();
  const Descriptor *DeclDesc = Ptr.getDeclDesc();
  assert(DeclDesc);

  if (!UseFieldDesc || DetermineForCompleteObject) {
    // Lower bound, so we can't fall back to this.
    if (ReportMinimum && !DetermineForCompleteObject)
      return false;

    // Can't read beyond the pointer decl desc.
    if (!UseFieldDesc && !ReportMinimum && DeclDesc->getType()->isPointerType())
      return false;
  } else {
    if (isUserWritingOffTheEnd(ASTCtx, Ptr.expand())) {
      // If we cannot determine the size of the initial allocation, then we
      // can't given an accurate upper-bound. However, we are still able to give
      // conservative lower-bounds for Type=3.
      if (Kind == 1)
        return false;
    }
  }

  const Descriptor *Desc = UseFieldDesc ? Ptr.getFieldDesc() : DeclDesc;
  assert(Desc);

  std::optional<unsigned> FullSize = computeFullDescSize(ASTCtx, Desc);
  if (!FullSize)
    return false;

  unsigned ByteOffset;
  if (UseFieldDesc) {
    if (Ptr.isBaseClass())
      ByteOffset = computePointerOffset(ASTCtx, Ptr.getBase()) -
                   computePointerOffset(ASTCtx, Ptr);
    else
      ByteOffset =
          computePointerOffset(ASTCtx, Ptr) -
          computePointerOffset(ASTCtx, Ptr.expand().atIndex(0).narrow());
  } else
    ByteOffset = computePointerOffset(ASTCtx, Ptr);

  assert(ByteOffset <= *FullSize);
  unsigned Result = *FullSize - ByteOffset;

  pushInteger(S, Result, Call->getType());
  return true;
}

static bool interp__builtin_is_within_lifetime(InterpState &S, CodePtr OpPC,
                                               const CallExpr *Call) {

  if (!S.inConstantContext())
    return false;

  const Pointer &Ptr = S.Stk.pop<Pointer>();

  auto Error = [&](int Diag) {
    bool CalledFromStd = false;
    const auto *Callee = S.Current->getCallee();
    if (Callee && Callee->isInStdNamespace()) {
      const IdentifierInfo *Identifier = Callee->getIdentifier();
      CalledFromStd = Identifier && Identifier->isStr("is_within_lifetime");
    }
    S.CCEDiag(CalledFromStd
                  ? S.Current->Caller->getSource(S.Current->getRetPC())
                  : S.Current->getSource(OpPC),
              diag::err_invalid_is_within_lifetime)
        << (CalledFromStd ? "std::is_within_lifetime"
                          : "__builtin_is_within_lifetime")
        << Diag;
    return false;
  };

  if (Ptr.isZero())
    return Error(0);
  if (Ptr.isOnePastEnd())
    return Error(1);

  bool Result = Ptr.getLifetime() != Lifetime::Ended;
  if (!Ptr.isActive()) {
    Result = false;
  } else {
    if (!CheckLive(S, OpPC, Ptr, AK_Read))
      return false;
    if (!CheckMutable(S, OpPC, Ptr))
      return false;
    if (!CheckDummy(S, OpPC, Ptr.block(), AK_Read))
      return false;
  }

  // Check if we're currently running an initializer.
  if (llvm::is_contained(S.InitializingBlocks, Ptr.block()))
    return Error(2);
  if (S.EvaluatingDecl && Ptr.getDeclDesc()->asVarDecl() == S.EvaluatingDecl)
    return Error(2);

  pushInteger(S, Result, Call->getType());
  return true;
}

static bool interp__builtin_elementwise_int_binop(
    InterpState &S, CodePtr OpPC, const CallExpr *Call, unsigned BuiltinID,
    llvm::function_ref<APInt(const APSInt &, const APSInt &)> Fn) {
  assert(Call->getNumArgs() == 2);

  // Single integer case.
  if (!Call->getArg(0)->getType()->isVectorType()) {
    assert(!Call->getArg(1)->getType()->isVectorType());
    APSInt RHS = popToAPSInt(
        S.Stk, *S.getContext().classify(Call->getArg(1)->getType()));
    APSInt LHS = popToAPSInt(
        S.Stk, *S.getContext().classify(Call->getArg(0)->getType()));
    APInt Result = Fn(LHS, RHS);
    pushInteger(S, APSInt(std::move(Result), !LHS.isSigned()), Call->getType());
    return true;
  }

  // Vector case.
  assert(Call->getArg(0)->getType()->isVectorType() &&
         Call->getArg(1)->getType()->isVectorType());
  const auto *VT = Call->getArg(0)->getType()->castAs<VectorType>();
  assert(VT->getElementType() ==
         Call->getArg(1)->getType()->castAs<VectorType>()->getElementType());
  assert(VT->getNumElements() ==
         Call->getArg(1)->getType()->castAs<VectorType>()->getNumElements());
  assert(VT->getElementType()->isIntegralOrEnumerationType());

  const Pointer &RHS = S.Stk.pop<Pointer>();
  const Pointer &LHS = S.Stk.pop<Pointer>();
  const Pointer &Dst = S.Stk.peek<Pointer>();
  PrimType ElemT = *S.getContext().classify(VT->getElementType());
  unsigned NumElems = VT->getNumElements();
  for (unsigned I = 0; I != NumElems; ++I) {
    APSInt Elem1;
    APSInt Elem2;
    INT_TYPE_SWITCH_NO_BOOL(ElemT, {
      Elem1 = LHS.elem<T>(I).toAPSInt();
      Elem2 = RHS.elem<T>(I).toAPSInt();
    });

    APSInt Result =
        APSInt(Fn(Elem1, Elem2),
               Call->getType()->isUnsignedIntegerOrEnumerationType());

    INT_TYPE_SWITCH_NO_BOOL(ElemT,
                            { Dst.elem<T>(I) = static_cast<T>(Result); });
  }
  Dst.initializeAllElements();

  return true;
}

static bool interp__builtin_elementwise_maxmin(InterpState &S, CodePtr OpPC,
                                               const CallExpr *Call,
                                               unsigned BuiltinID) {
  assert(Call->getNumArgs() == 2);

  QualType Arg0Type = Call->getArg(0)->getType();

  // TODO: Support floating-point types.
  if (!(Arg0Type->isIntegerType() ||
        (Arg0Type->isVectorType() &&
         Arg0Type->castAs<VectorType>()->getElementType()->isIntegerType())))
    return false;

  if (!Arg0Type->isVectorType()) {
    assert(!Call->getArg(1)->getType()->isVectorType());
    APSInt RHS = popToAPSInt(
        S.Stk, *S.getContext().classify(Call->getArg(1)->getType()));
    APSInt LHS = popToAPSInt(
        S.Stk, *S.getContext().classify(Call->getArg(0)->getType()));
    APInt Result;
    if (BuiltinID == Builtin::BI__builtin_elementwise_max) {
      Result = std::max(LHS, RHS);
    } else if (BuiltinID == Builtin::BI__builtin_elementwise_min) {
      Result = std::min(LHS, RHS);
    } else {
      llvm_unreachable("Wrong builtin ID");
    }

    pushInteger(S, APSInt(Result, !LHS.isSigned()), Call->getType());
    return true;
  }

  // Vector case.
  assert(Call->getArg(0)->getType()->isVectorType() &&
         Call->getArg(1)->getType()->isVectorType());
  const auto *VT = Call->getArg(0)->getType()->castAs<VectorType>();
  assert(VT->getElementType() ==
         Call->getArg(1)->getType()->castAs<VectorType>()->getElementType());
  assert(VT->getNumElements() ==
         Call->getArg(1)->getType()->castAs<VectorType>()->getNumElements());
  assert(VT->getElementType()->isIntegralOrEnumerationType());

  const Pointer &RHS = S.Stk.pop<Pointer>();
  const Pointer &LHS = S.Stk.pop<Pointer>();
  const Pointer &Dst = S.Stk.peek<Pointer>();
  PrimType ElemT = *S.getContext().classify(VT->getElementType());
  unsigned NumElems = VT->getNumElements();
  for (unsigned I = 0; I != NumElems; ++I) {
    APSInt Elem1;
    APSInt Elem2;
    INT_TYPE_SWITCH_NO_BOOL(ElemT, {
      Elem1 = LHS.elem<T>(I).toAPSInt();
      Elem2 = RHS.elem<T>(I).toAPSInt();
    });

    APSInt Result;
    if (BuiltinID == Builtin::BI__builtin_elementwise_max) {
      Result = APSInt(std::max(Elem1, Elem2),
                      Call->getType()->isUnsignedIntegerOrEnumerationType());
    } else if (BuiltinID == Builtin::BI__builtin_elementwise_min) {
      Result = APSInt(std::min(Elem1, Elem2),
                      Call->getType()->isUnsignedIntegerOrEnumerationType());
    } else {
      llvm_unreachable("Wrong builtin ID");
    }

    INT_TYPE_SWITCH_NO_BOOL(ElemT,
                            { Dst.elem<T>(I) = static_cast<T>(Result); });
  }
  Dst.initializeAllElements();

  return true;
}

static bool interp__builtin_ia32_pmul(InterpState &S, CodePtr OpPC,
                                      const CallExpr *Call,
                                      unsigned BuiltinID) {
  assert(Call->getArg(0)->getType()->isVectorType() &&
         Call->getArg(1)->getType()->isVectorType());
  const Pointer &RHS = S.Stk.pop<Pointer>();
  const Pointer &LHS = S.Stk.pop<Pointer>();
  const Pointer &Dst = S.Stk.peek<Pointer>();

  const auto *VT = Call->getArg(0)->getType()->castAs<VectorType>();
  PrimType ElemT = *S.getContext().classify(VT->getElementType());
  unsigned SourceLen = VT->getNumElements();

  PrimType DstElemT = *S.getContext().classify(
      Call->getType()->castAs<VectorType>()->getElementType());
  unsigned DstElem = 0;
  for (unsigned I = 0; I != SourceLen; I += 2) {
    APSInt Elem1;
    APSInt Elem2;
    INT_TYPE_SWITCH_NO_BOOL(ElemT, {
      Elem1 = LHS.elem<T>(I).toAPSInt();
      Elem2 = RHS.elem<T>(I).toAPSInt();
    });

    APSInt Result;
    switch (BuiltinID) {
    case clang::X86::BI__builtin_ia32_pmuludq128:
    case clang::X86::BI__builtin_ia32_pmuludq256:
    case clang::X86::BI__builtin_ia32_pmuludq512:
      Result = APSInt(llvm::APIntOps::muluExtended(Elem1, Elem2),
                      /*IsUnsigned=*/true);
      break;
    case clang::X86::BI__builtin_ia32_pmuldq128:
    case clang::X86::BI__builtin_ia32_pmuldq256:
    case clang::X86::BI__builtin_ia32_pmuldq512:
      Result = APSInt(llvm::APIntOps::mulsExtended(Elem1, Elem2),
                      /*IsUnsigned=*/false);
      break;
    }
    INT_TYPE_SWITCH_NO_BOOL(DstElemT,
                            { Dst.elem<T>(DstElem) = static_cast<T>(Result); });
    ++DstElem;
  }

  Dst.initializeAllElements();
  return true;
}

static bool interp__builtin_elementwise_fma(InterpState &S, CodePtr OpPC,
                                            const CallExpr *Call) {
  assert(Call->getNumArgs() == 3);

  FPOptions FPO = Call->getFPFeaturesInEffect(S.Ctx.getLangOpts());
  llvm::RoundingMode RM = getRoundingMode(FPO);
  const QualType Arg1Type = Call->getArg(0)->getType();
  const QualType Arg2Type = Call->getArg(1)->getType();
  const QualType Arg3Type = Call->getArg(2)->getType();

  // Non-vector floating point types.
  if (!Arg1Type->isVectorType()) {
    assert(!Arg2Type->isVectorType());
    assert(!Arg3Type->isVectorType());
    (void)Arg2Type;
    (void)Arg3Type;

    const Floating &Z = S.Stk.pop<Floating>();
    const Floating &Y = S.Stk.pop<Floating>();
    const Floating &X = S.Stk.pop<Floating>();
    APFloat F = X.getAPFloat();
    F.fusedMultiplyAdd(Y.getAPFloat(), Z.getAPFloat(), RM);
    Floating Result = S.allocFloat(X.getSemantics());
    Result.copy(F);
    S.Stk.push<Floating>(Result);
    return true;
  }

  // Vector type.
  assert(Arg1Type->isVectorType() && Arg2Type->isVectorType() &&
         Arg3Type->isVectorType());

  const VectorType *VecT = Arg1Type->castAs<VectorType>();
  const QualType ElemT = VecT->getElementType();
  unsigned NumElems = VecT->getNumElements();

  assert(ElemT == Arg2Type->castAs<VectorType>()->getElementType() &&
         ElemT == Arg3Type->castAs<VectorType>()->getElementType());
  assert(NumElems == Arg2Type->castAs<VectorType>()->getNumElements() &&
         NumElems == Arg3Type->castAs<VectorType>()->getNumElements());
  assert(ElemT->isRealFloatingType());
  (void)ElemT;

  const Pointer &VZ = S.Stk.pop<Pointer>();
  const Pointer &VY = S.Stk.pop<Pointer>();
  const Pointer &VX = S.Stk.pop<Pointer>();
  const Pointer &Dst = S.Stk.peek<Pointer>();
  for (unsigned I = 0; I != NumElems; ++I) {
    using T = PrimConv<PT_Float>::T;
    APFloat X = VX.elem<T>(I).getAPFloat();
    APFloat Y = VY.elem<T>(I).getAPFloat();
    APFloat Z = VZ.elem<T>(I).getAPFloat();
    (void)X.fusedMultiplyAdd(Y, Z, RM);
    Dst.elem<Floating>(I) = Floating(X);
  }
  Dst.initializeAllElements();
  return true;
}

/// AVX512 predicated move: "Result = Mask[] ? LHS[] : RHS[]".
static bool interp__builtin_select(InterpState &S, CodePtr OpPC,
                                   const CallExpr *Call) {
  const Pointer &RHS = S.Stk.pop<Pointer>();
  const Pointer &LHS = S.Stk.pop<Pointer>();
  PrimType MaskT = *S.getContext().classify(Call->getArg(0));
  APSInt Mask = popToAPSInt(S.Stk, MaskT);
  const Pointer &Dst = S.Stk.peek<Pointer>();

  assert(LHS.getNumElems() == RHS.getNumElems());
  assert(LHS.getNumElems() == Dst.getNumElems());
  unsigned NumElems = LHS.getNumElems();
  PrimType ElemT = LHS.getFieldDesc()->getPrimType();
  PrimType DstElemT = Dst.getFieldDesc()->getPrimType();

  for (unsigned I = 0; I != NumElems; ++I) {
    if (ElemT == PT_Float) {
      assert(DstElemT == PT_Float);
      Dst.elem<Floating>(I) =
          Mask[I] ? LHS.elem<Floating>(I) : RHS.elem<Floating>(I);
    } else {
      APSInt Elem;
      INT_TYPE_SWITCH(ElemT, {
        Elem = Mask[I] ? LHS.elem<T>(I).toAPSInt() : RHS.elem<T>(I).toAPSInt();
      });
      INT_TYPE_SWITCH_NO_BOOL(DstElemT,
                              { Dst.elem<T>(I) = static_cast<T>(Elem); });
    }
  }
  Dst.initializeAllElements();

  return true;
}

bool InterpretBuiltin(InterpState &S, CodePtr OpPC, const CallExpr *Call,
                      uint32_t BuiltinID) {
  if (!S.getASTContext().BuiltinInfo.isConstantEvaluated(BuiltinID))
    return Invalid(S, OpPC);

  const InterpFrame *Frame = S.Current;
  switch (BuiltinID) {
  case Builtin::BI__builtin_is_constant_evaluated:
    return interp__builtin_is_constant_evaluated(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_assume:
  case Builtin::BI__assume:
    return interp__builtin_assume(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_strcmp:
  case Builtin::BIstrcmp:
  case Builtin::BI__builtin_strncmp:
  case Builtin::BIstrncmp:
  case Builtin::BI__builtin_wcsncmp:
  case Builtin::BIwcsncmp:
  case Builtin::BI__builtin_wcscmp:
  case Builtin::BIwcscmp:
    return interp__builtin_strcmp(S, OpPC, Frame, Call, BuiltinID);

  case Builtin::BI__builtin_strlen:
  case Builtin::BIstrlen:
  case Builtin::BI__builtin_wcslen:
  case Builtin::BIwcslen:
    return interp__builtin_strlen(S, OpPC, Frame, Call, BuiltinID);

  case Builtin::BI__builtin_nan:
  case Builtin::BI__builtin_nanf:
  case Builtin::BI__builtin_nanl:
  case Builtin::BI__builtin_nanf16:
  case Builtin::BI__builtin_nanf128:
    return interp__builtin_nan(S, OpPC, Frame, Call, /*Signaling=*/false);

  case Builtin::BI__builtin_nans:
  case Builtin::BI__builtin_nansf:
  case Builtin::BI__builtin_nansl:
  case Builtin::BI__builtin_nansf16:
  case Builtin::BI__builtin_nansf128:
    return interp__builtin_nan(S, OpPC, Frame, Call, /*Signaling=*/true);

  case Builtin::BI__builtin_huge_val:
  case Builtin::BI__builtin_huge_valf:
  case Builtin::BI__builtin_huge_vall:
  case Builtin::BI__builtin_huge_valf16:
  case Builtin::BI__builtin_huge_valf128:
  case Builtin::BI__builtin_inf:
  case Builtin::BI__builtin_inff:
  case Builtin::BI__builtin_infl:
  case Builtin::BI__builtin_inff16:
  case Builtin::BI__builtin_inff128:
    return interp__builtin_inf(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_copysign:
  case Builtin::BI__builtin_copysignf:
  case Builtin::BI__builtin_copysignl:
  case Builtin::BI__builtin_copysignf128:
    return interp__builtin_copysign(S, OpPC, Frame);

  case Builtin::BI__builtin_fmin:
  case Builtin::BI__builtin_fminf:
  case Builtin::BI__builtin_fminl:
  case Builtin::BI__builtin_fminf16:
  case Builtin::BI__builtin_fminf128:
    return interp__builtin_fmin(S, OpPC, Frame, /*IsNumBuiltin=*/false);

  case Builtin::BI__builtin_fminimum_num:
  case Builtin::BI__builtin_fminimum_numf:
  case Builtin::BI__builtin_fminimum_numl:
  case Builtin::BI__builtin_fminimum_numf16:
  case Builtin::BI__builtin_fminimum_numf128:
    return interp__builtin_fmin(S, OpPC, Frame, /*IsNumBuiltin=*/true);

  case Builtin::BI__builtin_fmax:
  case Builtin::BI__builtin_fmaxf:
  case Builtin::BI__builtin_fmaxl:
  case Builtin::BI__builtin_fmaxf16:
  case Builtin::BI__builtin_fmaxf128:
    return interp__builtin_fmax(S, OpPC, Frame, /*IsNumBuiltin=*/false);

  case Builtin::BI__builtin_fmaximum_num:
  case Builtin::BI__builtin_fmaximum_numf:
  case Builtin::BI__builtin_fmaximum_numl:
  case Builtin::BI__builtin_fmaximum_numf16:
  case Builtin::BI__builtin_fmaximum_numf128:
    return interp__builtin_fmax(S, OpPC, Frame, /*IsNumBuiltin=*/true);

  case Builtin::BI__builtin_isnan:
    return interp__builtin_isnan(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_issignaling:
    return interp__builtin_issignaling(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_isinf:
    return interp__builtin_isinf(S, OpPC, Frame, /*Sign=*/false, Call);

  case Builtin::BI__builtin_isinf_sign:
    return interp__builtin_isinf(S, OpPC, Frame, /*Sign=*/true, Call);

  case Builtin::BI__builtin_isfinite:
    return interp__builtin_isfinite(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_isnormal:
    return interp__builtin_isnormal(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_issubnormal:
    return interp__builtin_issubnormal(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_iszero:
    return interp__builtin_iszero(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_signbit:
  case Builtin::BI__builtin_signbitf:
  case Builtin::BI__builtin_signbitl:
    return interp__builtin_signbit(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_isgreater:
  case Builtin::BI__builtin_isgreaterequal:
  case Builtin::BI__builtin_isless:
  case Builtin::BI__builtin_islessequal:
  case Builtin::BI__builtin_islessgreater:
  case Builtin::BI__builtin_isunordered:
    return interp_floating_comparison(S, OpPC, Call, BuiltinID);

  case Builtin::BI__builtin_isfpclass:
    return interp__builtin_isfpclass(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_fpclassify:
    return interp__builtin_fpclassify(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_fabs:
  case Builtin::BI__builtin_fabsf:
  case Builtin::BI__builtin_fabsl:
  case Builtin::BI__builtin_fabsf128:
    return interp__builtin_fabs(S, OpPC, Frame);

  case Builtin::BI__builtin_abs:
  case Builtin::BI__builtin_labs:
  case Builtin::BI__builtin_llabs:
    return interp__builtin_abs(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll:
  case Builtin::BI__builtin_popcountg:
  case Builtin::BI__popcnt16: // Microsoft variants of popcount
  case Builtin::BI__popcnt:
  case Builtin::BI__popcnt64:
    return interp__builtin_popcount(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_parity:
  case Builtin::BI__builtin_parityl:
  case Builtin::BI__builtin_parityll:
    return interp__builtin_parity(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_clrsb:
  case Builtin::BI__builtin_clrsbl:
  case Builtin::BI__builtin_clrsbll:
    return interp__builtin_clrsb(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_bitreverse8:
  case Builtin::BI__builtin_bitreverse16:
  case Builtin::BI__builtin_bitreverse32:
  case Builtin::BI__builtin_bitreverse64:
    return interp__builtin_bitreverse(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_classify_type:
    return interp__builtin_classify_type(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_expect:
  case Builtin::BI__builtin_expect_with_probability:
    return interp__builtin_expect(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_rotateleft8:
  case Builtin::BI__builtin_rotateleft16:
  case Builtin::BI__builtin_rotateleft32:
  case Builtin::BI__builtin_rotateleft64:
  case Builtin::BI_rotl8: // Microsoft variants of rotate left
  case Builtin::BI_rotl16:
  case Builtin::BI_rotl:
  case Builtin::BI_lrotl:
  case Builtin::BI_rotl64:
    return interp__builtin_rotate(S, OpPC, Frame, Call, /*Right=*/false);

  case Builtin::BI__builtin_rotateright8:
  case Builtin::BI__builtin_rotateright16:
  case Builtin::BI__builtin_rotateright32:
  case Builtin::BI__builtin_rotateright64:
  case Builtin::BI_rotr8: // Microsoft variants of rotate right
  case Builtin::BI_rotr16:
  case Builtin::BI_rotr:
  case Builtin::BI_lrotr:
  case Builtin::BI_rotr64:
    return interp__builtin_rotate(S, OpPC, Frame, Call, /*Right=*/true);

  case Builtin::BI__builtin_ffs:
  case Builtin::BI__builtin_ffsl:
  case Builtin::BI__builtin_ffsll:
    return interp__builtin_ffs(S, OpPC, Frame, Call);

  case Builtin::BIaddressof:
  case Builtin::BI__addressof:
  case Builtin::BI__builtin_addressof:
    assert(isNoopBuiltin(BuiltinID));
    return interp__builtin_addressof(S, OpPC, Frame, Call);

  case Builtin::BIas_const:
  case Builtin::BIforward:
  case Builtin::BIforward_like:
  case Builtin::BImove:
  case Builtin::BImove_if_noexcept:
    assert(isNoopBuiltin(BuiltinID));
    return interp__builtin_move(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_eh_return_data_regno:
    return interp__builtin_eh_return_data_regno(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_launder:
    assert(isNoopBuiltin(BuiltinID));
    return true;

  case Builtin::BI__builtin_add_overflow:
  case Builtin::BI__builtin_sub_overflow:
  case Builtin::BI__builtin_mul_overflow:
  case Builtin::BI__builtin_sadd_overflow:
  case Builtin::BI__builtin_uadd_overflow:
  case Builtin::BI__builtin_uaddl_overflow:
  case Builtin::BI__builtin_uaddll_overflow:
  case Builtin::BI__builtin_usub_overflow:
  case Builtin::BI__builtin_usubl_overflow:
  case Builtin::BI__builtin_usubll_overflow:
  case Builtin::BI__builtin_umul_overflow:
  case Builtin::BI__builtin_umull_overflow:
  case Builtin::BI__builtin_umulll_overflow:
  case Builtin::BI__builtin_saddl_overflow:
  case Builtin::BI__builtin_saddll_overflow:
  case Builtin::BI__builtin_ssub_overflow:
  case Builtin::BI__builtin_ssubl_overflow:
  case Builtin::BI__builtin_ssubll_overflow:
  case Builtin::BI__builtin_smul_overflow:
  case Builtin::BI__builtin_smull_overflow:
  case Builtin::BI__builtin_smulll_overflow:
    return interp__builtin_overflowop(S, OpPC, Call, BuiltinID);

  case Builtin::BI__builtin_addcb:
  case Builtin::BI__builtin_addcs:
  case Builtin::BI__builtin_addc:
  case Builtin::BI__builtin_addcl:
  case Builtin::BI__builtin_addcll:
  case Builtin::BI__builtin_subcb:
  case Builtin::BI__builtin_subcs:
  case Builtin::BI__builtin_subc:
  case Builtin::BI__builtin_subcl:
  case Builtin::BI__builtin_subcll:
    return interp__builtin_carryop(S, OpPC, Frame, Call, BuiltinID);

  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll:
  case Builtin::BI__builtin_clzs:
  case Builtin::BI__builtin_clzg:
  case Builtin::BI__lzcnt16: // Microsoft variants of count leading-zeroes
  case Builtin::BI__lzcnt:
  case Builtin::BI__lzcnt64:
    return interp__builtin_clz(S, OpPC, Frame, Call, BuiltinID);

  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll:
  case Builtin::BI__builtin_ctzs:
  case Builtin::BI__builtin_ctzg:
    return interp__builtin_ctz(S, OpPC, Frame, Call, BuiltinID);

  case Builtin::BI__builtin_elementwise_ctlz:
  case Builtin::BI__builtin_elementwise_cttz:
    return interp__builtin_elementwise_countzeroes(S, OpPC, Frame, Call,
                                                   BuiltinID);

  case Builtin::BI__builtin_bswap16:
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64:
    return interp__builtin_bswap(S, OpPC, Frame, Call);

  case Builtin::BI__atomic_always_lock_free:
  case Builtin::BI__atomic_is_lock_free:
    return interp__builtin_atomic_lock_free(S, OpPC, Frame, Call, BuiltinID);

  case Builtin::BI__c11_atomic_is_lock_free:
    return interp__builtin_c11_atomic_is_lock_free(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_complex:
    return interp__builtin_complex(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_is_aligned:
  case Builtin::BI__builtin_align_up:
  case Builtin::BI__builtin_align_down:
    return interp__builtin_is_aligned_up_down(S, OpPC, Frame, Call, BuiltinID);

  case Builtin::BI__builtin_assume_aligned:
    return interp__builtin_assume_aligned(S, OpPC, Frame, Call);

  case clang::X86::BI__builtin_ia32_bextr_u32:
  case clang::X86::BI__builtin_ia32_bextr_u64:
  case clang::X86::BI__builtin_ia32_bextri_u32:
  case clang::X86::BI__builtin_ia32_bextri_u64:
    return interp__builtin_ia32_bextr(S, OpPC, Frame, Call);

  case clang::X86::BI__builtin_ia32_bzhi_si:
  case clang::X86::BI__builtin_ia32_bzhi_di:
    return interp__builtin_ia32_bzhi(S, OpPC, Frame, Call);

  case clang::X86::BI__builtin_ia32_lzcnt_u16:
  case clang::X86::BI__builtin_ia32_lzcnt_u32:
  case clang::X86::BI__builtin_ia32_lzcnt_u64:
    return interp__builtin_ia32_lzcnt(S, OpPC, Frame, Call);

  case clang::X86::BI__builtin_ia32_tzcnt_u16:
  case clang::X86::BI__builtin_ia32_tzcnt_u32:
  case clang::X86::BI__builtin_ia32_tzcnt_u64:
    return interp__builtin_ia32_tzcnt(S, OpPC, Frame, Call);

  case clang::X86::BI__builtin_ia32_pdep_si:
  case clang::X86::BI__builtin_ia32_pdep_di:
    return interp__builtin_ia32_pdep(S, OpPC, Frame, Call);

  case clang::X86::BI__builtin_ia32_pext_si:
  case clang::X86::BI__builtin_ia32_pext_di:
    return interp__builtin_ia32_pext(S, OpPC, Frame, Call);

  case clang::X86::BI__builtin_ia32_addcarryx_u32:
  case clang::X86::BI__builtin_ia32_addcarryx_u64:
  case clang::X86::BI__builtin_ia32_subborrow_u32:
  case clang::X86::BI__builtin_ia32_subborrow_u64:
    return interp__builtin_ia32_addcarry_subborrow(S, OpPC, Frame, Call,
                                                   BuiltinID);

  case Builtin::BI__builtin_os_log_format_buffer_size:
    return interp__builtin_os_log_format_buffer_size(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_ptrauth_string_discriminator:
    return interp__builtin_ptrauth_string_discriminator(S, OpPC, Frame, Call);

  case Builtin::BI__noop:
    pushInteger(S, 0, Call->getType());
    return true;

  case Builtin::BI__builtin_operator_new:
    return interp__builtin_operator_new(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_operator_delete:
    return interp__builtin_operator_delete(S, OpPC, Frame, Call);

  case Builtin::BI__arithmetic_fence:
    return interp__builtin_arithmetic_fence(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_reduce_add:
  case Builtin::BI__builtin_reduce_mul:
  case Builtin::BI__builtin_reduce_and:
  case Builtin::BI__builtin_reduce_or:
  case Builtin::BI__builtin_reduce_xor:
  case Builtin::BI__builtin_reduce_min:
  case Builtin::BI__builtin_reduce_max:
    return interp__builtin_vector_reduce(S, OpPC, Call, BuiltinID);

  case Builtin::BI__builtin_elementwise_popcount:
  case Builtin::BI__builtin_elementwise_bitreverse:
    return interp__builtin_elementwise_popcount(S, OpPC, Frame, Call,
                                                BuiltinID);

  case Builtin::BI__builtin_elementwise_abs:
    return interp__builtin_elementwise_abs(S, OpPC, Frame, Call, BuiltinID);

  case Builtin::BI__builtin_memcpy:
  case Builtin::BImemcpy:
  case Builtin::BI__builtin_wmemcpy:
  case Builtin::BIwmemcpy:
  case Builtin::BI__builtin_memmove:
  case Builtin::BImemmove:
  case Builtin::BI__builtin_wmemmove:
  case Builtin::BIwmemmove:
    return interp__builtin_memcpy(S, OpPC, Frame, Call, BuiltinID);

  case Builtin::BI__builtin_memcmp:
  case Builtin::BImemcmp:
  case Builtin::BI__builtin_bcmp:
  case Builtin::BIbcmp:
  case Builtin::BI__builtin_wmemcmp:
  case Builtin::BIwmemcmp:
    return interp__builtin_memcmp(S, OpPC, Frame, Call, BuiltinID);

  case Builtin::BImemchr:
  case Builtin::BI__builtin_memchr:
  case Builtin::BIstrchr:
  case Builtin::BI__builtin_strchr:
  case Builtin::BIwmemchr:
  case Builtin::BI__builtin_wmemchr:
  case Builtin::BIwcschr:
  case Builtin::BI__builtin_wcschr:
  case Builtin::BI__builtin_char_memchr:
    return interp__builtin_memchr(S, OpPC, Call, BuiltinID);

  case Builtin::BI__builtin_object_size:
  case Builtin::BI__builtin_dynamic_object_size:
    return interp__builtin_object_size(S, OpPC, Frame, Call);

  case Builtin::BI__builtin_is_within_lifetime:
    return interp__builtin_is_within_lifetime(S, OpPC, Call);

  case Builtin::BI__builtin_elementwise_add_sat:
    return interp__builtin_elementwise_int_binop(
        S, OpPC, Call, BuiltinID, [](const APSInt &LHS, const APSInt &RHS) {
          return LHS.isSigned() ? LHS.sadd_sat(RHS) : LHS.uadd_sat(RHS);
        });

  case Builtin::BI__builtin_elementwise_sub_sat:
    return interp__builtin_elementwise_int_binop(
        S, OpPC, Call, BuiltinID, [](const APSInt &LHS, const APSInt &RHS) {
          return LHS.isSigned() ? LHS.ssub_sat(RHS) : LHS.usub_sat(RHS);
        });

  case clang::X86::BI__builtin_ia32_pmulhuw128:
  case clang::X86::BI__builtin_ia32_pmulhuw256:
  case clang::X86::BI__builtin_ia32_pmulhuw512:
    return interp__builtin_elementwise_int_binop(S, OpPC, Call, BuiltinID,
                                                 llvm::APIntOps::mulhu);

  case clang::X86::BI__builtin_ia32_pmulhw128:
  case clang::X86::BI__builtin_ia32_pmulhw256:
  case clang::X86::BI__builtin_ia32_pmulhw512:
    return interp__builtin_elementwise_int_binop(S, OpPC, Call, BuiltinID,
                                                 llvm::APIntOps::mulhs);

  case clang::X86::BI__builtin_ia32_psllv2di:
  case clang::X86::BI__builtin_ia32_psllv4di:
  case clang::X86::BI__builtin_ia32_psllv4si:
  case clang::X86::BI__builtin_ia32_psllv8si:
    return interp__builtin_elementwise_int_binop(
        S, OpPC, Call, BuiltinID, [](const APSInt &LHS, const APSInt &RHS) {
          if (RHS.uge(RHS.getBitWidth())) {
            return APInt::getZero(RHS.getBitWidth());
          }
          return LHS.shl(RHS.getZExtValue());
        });

  case clang::X86::BI__builtin_ia32_psrav4si:
  case clang::X86::BI__builtin_ia32_psrav8si:
    return interp__builtin_elementwise_int_binop(
        S, OpPC, Call, BuiltinID, [](const APSInt &LHS, const APSInt &RHS) {
          if (RHS.uge(RHS.getBitWidth())) {
            return LHS.ashr(RHS.getBitWidth() - 1);
          }
          return LHS.ashr(RHS.getZExtValue());
        });

  case clang::X86::BI__builtin_ia32_psrlv2di:
  case clang::X86::BI__builtin_ia32_psrlv4di:
  case clang::X86::BI__builtin_ia32_psrlv4si:
  case clang::X86::BI__builtin_ia32_psrlv8si:
    return interp__builtin_elementwise_int_binop(
        S, OpPC, Call, BuiltinID, [](const APSInt &LHS, const APSInt &RHS) {
          if (RHS.uge(RHS.getBitWidth())) {
            return APInt::getZero(RHS.getBitWidth());
          }
          return LHS.lshr(RHS.getZExtValue());
        });

  case Builtin::BI__builtin_elementwise_max:
  case Builtin::BI__builtin_elementwise_min:
    return interp__builtin_elementwise_maxmin(S, OpPC, Call, BuiltinID);

  case clang::X86::BI__builtin_ia32_pmuldq128:
  case clang::X86::BI__builtin_ia32_pmuldq256:
  case clang::X86::BI__builtin_ia32_pmuldq512:
  case clang::X86::BI__builtin_ia32_pmuludq128:
  case clang::X86::BI__builtin_ia32_pmuludq256:
  case clang::X86::BI__builtin_ia32_pmuludq512:
    return interp__builtin_ia32_pmul(S, OpPC, Call, BuiltinID);

  case Builtin::BI__builtin_elementwise_fma:
    return interp__builtin_elementwise_fma(S, OpPC, Call);

  case X86::BI__builtin_ia32_selectb_128:
  case X86::BI__builtin_ia32_selectb_256:
  case X86::BI__builtin_ia32_selectb_512:
  case X86::BI__builtin_ia32_selectw_128:
  case X86::BI__builtin_ia32_selectw_256:
  case X86::BI__builtin_ia32_selectw_512:
  case X86::BI__builtin_ia32_selectd_128:
  case X86::BI__builtin_ia32_selectd_256:
  case X86::BI__builtin_ia32_selectd_512:
  case X86::BI__builtin_ia32_selectq_128:
  case X86::BI__builtin_ia32_selectq_256:
  case X86::BI__builtin_ia32_selectq_512:
  case X86::BI__builtin_ia32_selectph_128:
  case X86::BI__builtin_ia32_selectph_256:
  case X86::BI__builtin_ia32_selectph_512:
  case X86::BI__builtin_ia32_selectpbf_128:
  case X86::BI__builtin_ia32_selectpbf_256:
  case X86::BI__builtin_ia32_selectpbf_512:
  case X86::BI__builtin_ia32_selectps_128:
  case X86::BI__builtin_ia32_selectps_256:
  case X86::BI__builtin_ia32_selectps_512:
  case X86::BI__builtin_ia32_selectpd_128:
  case X86::BI__builtin_ia32_selectpd_256:
  case X86::BI__builtin_ia32_selectpd_512:
    return interp__builtin_select(S, OpPC, Call);

  default:
    S.FFDiag(S.Current->getLocation(OpPC),
             diag::note_invalid_subexpr_in_const_expr)
        << S.Current->getRange(OpPC);

    return false;
  }

  llvm_unreachable("Unhandled builtin ID");
}

bool InterpretOffsetOf(InterpState &S, CodePtr OpPC, const OffsetOfExpr *E,
                       ArrayRef<int64_t> ArrayIndices, int64_t &IntResult) {
  CharUnits Result;
  unsigned N = E->getNumComponents();
  assert(N > 0);

  unsigned ArrayIndex = 0;
  QualType CurrentType = E->getTypeSourceInfo()->getType();
  for (unsigned I = 0; I != N; ++I) {
    const OffsetOfNode &Node = E->getComponent(I);
    switch (Node.getKind()) {
    case OffsetOfNode::Field: {
      const FieldDecl *MemberDecl = Node.getField();
      const auto *RD = CurrentType->getAsRecordDecl();
      if (!RD || RD->isInvalidDecl())
        return false;
      const ASTRecordLayout &RL = S.getASTContext().getASTRecordLayout(RD);
      unsigned FieldIndex = MemberDecl->getFieldIndex();
      assert(FieldIndex < RL.getFieldCount() && "offsetof field in wrong type");
      Result +=
          S.getASTContext().toCharUnitsFromBits(RL.getFieldOffset(FieldIndex));
      CurrentType = MemberDecl->getType().getNonReferenceType();
      break;
    }
    case OffsetOfNode::Array: {
      // When generating bytecode, we put all the index expressions as Sint64 on
      // the stack.
      int64_t Index = ArrayIndices[ArrayIndex];
      const ArrayType *AT = S.getASTContext().getAsArrayType(CurrentType);
      if (!AT)
        return false;
      CurrentType = AT->getElementType();
      CharUnits ElementSize = S.getASTContext().getTypeSizeInChars(CurrentType);
      Result += Index * ElementSize;
      ++ArrayIndex;
      break;
    }
    case OffsetOfNode::Base: {
      const CXXBaseSpecifier *BaseSpec = Node.getBase();
      if (BaseSpec->isVirtual())
        return false;

      // Find the layout of the class whose base we are looking into.
      const auto *RD = CurrentType->getAsCXXRecordDecl();
      if (!RD || RD->isInvalidDecl())
        return false;
      const ASTRecordLayout &RL = S.getASTContext().getASTRecordLayout(RD);

      // Find the base class itself.
      CurrentType = BaseSpec->getType();
      const auto *BaseRD = CurrentType->getAsCXXRecordDecl();
      if (!BaseRD)
        return false;

      // Add the offset to the base.
      Result += RL.getBaseClassOffset(BaseRD);
      break;
    }
    case OffsetOfNode::Identifier:
      llvm_unreachable("Dependent OffsetOfExpr?");
    }
  }

  IntResult = Result.getQuantity();

  return true;
}

bool SetThreeWayComparisonField(InterpState &S, CodePtr OpPC,
                                const Pointer &Ptr, const APSInt &IntValue) {

  const Record *R = Ptr.getRecord();
  assert(R);
  assert(R->getNumFields() == 1);

  unsigned FieldOffset = R->getField(0u)->Offset;
  const Pointer &FieldPtr = Ptr.atField(FieldOffset);
  PrimType FieldT = *S.getContext().classify(FieldPtr.getType());

  INT_TYPE_SWITCH(FieldT,
                  FieldPtr.deref<T>() = T::from(IntValue.getSExtValue()));
  FieldPtr.initialize();
  return true;
}

static void zeroAll(Pointer &Dest) {
  const Descriptor *Desc = Dest.getFieldDesc();

  if (Desc->isPrimitive()) {
    TYPE_SWITCH(Desc->getPrimType(), {
      Dest.deref<T>().~T();
      new (&Dest.deref<T>()) T();
    });
    return;
  }

  if (Desc->isRecord()) {
    const Record *R = Desc->ElemRecord;
    for (const Record::Field &F : R->fields()) {
      Pointer FieldPtr = Dest.atField(F.Offset);
      zeroAll(FieldPtr);
    }
    return;
  }

  if (Desc->isPrimitiveArray()) {
    for (unsigned I = 0, N = Desc->getNumElems(); I != N; ++I) {
      TYPE_SWITCH(Desc->getPrimType(), {
        Dest.deref<T>().~T();
        new (&Dest.deref<T>()) T();
      });
    }
    return;
  }

  if (Desc->isCompositeArray()) {
    for (unsigned I = 0, N = Desc->getNumElems(); I != N; ++I) {
      Pointer ElemPtr = Dest.atIndex(I).narrow();
      zeroAll(ElemPtr);
    }
    return;
  }
}

static bool copyComposite(InterpState &S, CodePtr OpPC, const Pointer &Src,
                          Pointer &Dest, bool Activate);
static bool copyRecord(InterpState &S, CodePtr OpPC, const Pointer &Src,
                       Pointer &Dest, bool Activate = false) {
  [[maybe_unused]] const Descriptor *SrcDesc = Src.getFieldDesc();
  const Descriptor *DestDesc = Dest.getFieldDesc();

  auto copyField = [&](const Record::Field &F, bool Activate) -> bool {
    Pointer DestField = Dest.atField(F.Offset);
    if (OptPrimType FT = S.Ctx.classify(F.Decl->getType())) {
      TYPE_SWITCH(*FT, {
        DestField.deref<T>() = Src.atField(F.Offset).deref<T>();
        if (Src.atField(F.Offset).isInitialized())
          DestField.initialize();
        if (Activate)
          DestField.activate();
      });
      return true;
    }
    // Composite field.
    return copyComposite(S, OpPC, Src.atField(F.Offset), DestField, Activate);
  };

  assert(SrcDesc->isRecord());
  assert(SrcDesc->ElemRecord == DestDesc->ElemRecord);
  const Record *R = DestDesc->ElemRecord;
  for (const Record::Field &F : R->fields()) {
    if (R->isUnion()) {
      // For unions, only copy the active field. Zero all others.
      const Pointer &SrcField = Src.atField(F.Offset);
      if (SrcField.isActive()) {
        if (!copyField(F, /*Activate=*/true))
          return false;
      } else {
        if (!CheckMutable(S, OpPC, Src.atField(F.Offset)))
          return false;
        Pointer DestField = Dest.atField(F.Offset);
        zeroAll(DestField);
      }
    } else {
      if (!copyField(F, Activate))
        return false;
    }
  }

  for (const Record::Base &B : R->bases()) {
    Pointer DestBase = Dest.atField(B.Offset);
    if (!copyRecord(S, OpPC, Src.atField(B.Offset), DestBase, Activate))
      return false;
  }

  Dest.initialize();
  return true;
}

static bool copyComposite(InterpState &S, CodePtr OpPC, const Pointer &Src,
                          Pointer &Dest, bool Activate = false) {
  assert(Src.isLive() && Dest.isLive());

  [[maybe_unused]] const Descriptor *SrcDesc = Src.getFieldDesc();
  const Descriptor *DestDesc = Dest.getFieldDesc();

  assert(!DestDesc->isPrimitive() && !SrcDesc->isPrimitive());

  if (DestDesc->isPrimitiveArray()) {
    assert(SrcDesc->isPrimitiveArray());
    assert(SrcDesc->getNumElems() == DestDesc->getNumElems());
    PrimType ET = DestDesc->getPrimType();
    for (unsigned I = 0, N = DestDesc->getNumElems(); I != N; ++I) {
      Pointer DestElem = Dest.atIndex(I);
      TYPE_SWITCH(ET, {
        DestElem.deref<T>() = Src.elem<T>(I);
        DestElem.initialize();
      });
    }
    return true;
  }

  if (DestDesc->isCompositeArray()) {
    assert(SrcDesc->isCompositeArray());
    assert(SrcDesc->getNumElems() == DestDesc->getNumElems());
    for (unsigned I = 0, N = DestDesc->getNumElems(); I != N; ++I) {
      const Pointer &SrcElem = Src.atIndex(I).narrow();
      Pointer DestElem = Dest.atIndex(I).narrow();
      if (!copyComposite(S, OpPC, SrcElem, DestElem, Activate))
        return false;
    }
    return true;
  }

  if (DestDesc->isRecord())
    return copyRecord(S, OpPC, Src, Dest, Activate);
  return Invalid(S, OpPC);
}

bool DoMemcpy(InterpState &S, CodePtr OpPC, const Pointer &Src, Pointer &Dest) {
  return copyComposite(S, OpPC, Src, Dest);
}

} // namespace interp
} // namespace clang
