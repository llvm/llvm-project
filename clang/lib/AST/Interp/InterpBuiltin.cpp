//===--- InterpBuiltin.cpp - Interpreter for the constexpr VM ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../ExprConstShared.h"
#include "Boolean.h"
#include "Interp.h"
#include "PrimType.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetInfo.h"

namespace clang {
namespace interp {

template <typename T>
static T getParam(const InterpFrame *Frame, unsigned Index) {
  assert(Frame->getFunction()->getNumParams() > Index);
  unsigned Offset = Frame->getFunction()->getParamOffset(Index);
  return Frame->getParam<T>(Offset);
}

PrimType getIntPrimType(const InterpState &S) {
  const TargetInfo &TI = S.getCtx().getTargetInfo();
  unsigned IntWidth = TI.getIntWidth();

  if (IntWidth == 32)
    return PT_Sint32;
  else if (IntWidth == 16)
    return PT_Sint16;
  llvm_unreachable("Int isn't 16 or 32 bit?");
}

PrimType getLongPrimType(const InterpState &S) {
  const TargetInfo &TI = S.getCtx().getTargetInfo();
  unsigned LongWidth = TI.getLongWidth();

  if (LongWidth == 64)
    return PT_Sint64;
  else if (LongWidth == 32)
    return PT_Sint32;
  else if (LongWidth == 16)
    return PT_Sint16;
  llvm_unreachable("long isn't 16, 32 or 64 bit?");
}

/// Peek an integer value from the stack into an APSInt.
static APSInt peekToAPSInt(InterpStack &Stk, PrimType T, size_t Offset = 0) {
  if (Offset == 0)
    Offset = align(primSize(T));

  APSInt R;
  INT_TYPE_SWITCH(T, {
    T Val = Stk.peek<T>(Offset);
    R = APSInt(
        APInt(Val.bitWidth(), static_cast<uint64_t>(Val), T::isSigned()));
  });

  return R;
}

/// Pushes \p Val to the stack, as a target-dependent 'int'.
static void pushInt(InterpState &S, int32_t Val) {
  PrimType IntType = getIntPrimType(S);
  if (IntType == PT_Sint32)
    S.Stk.push<Integral<32, true>>(Integral<32, true>::from(Val));
  else if (IntType == PT_Sint16)
    S.Stk.push<Integral<16, true>>(Integral<16, true>::from(Val));
  else
    llvm_unreachable("Int isn't 16 or 32 bit?");
}

static void pushAPSInt(InterpState &S, const APSInt &Val) {
  bool Signed = Val.isSigned();

  if (Signed) {
    switch (Val.getBitWidth()) {
    case 64:
      S.Stk.push<Integral<64, true>>(
          Integral<64, true>::from(Val.getSExtValue()));
      break;
    case 32:
      S.Stk.push<Integral<32, true>>(
          Integral<32, true>::from(Val.getSExtValue()));
      break;
    case 16:
      S.Stk.push<Integral<16, true>>(
          Integral<16, true>::from(Val.getSExtValue()));
      break;
    case 8:
      S.Stk.push<Integral<8, true>>(
          Integral<8, true>::from(Val.getSExtValue()));
      break;
    default:
      llvm_unreachable("Invalid integer bitwidth");
    }
    return;
  }

  // Unsigned.
  switch (Val.getBitWidth()) {
  case 64:
    S.Stk.push<Integral<64, false>>(
        Integral<64, false>::from(Val.getZExtValue()));
    break;
  case 32:
    S.Stk.push<Integral<32, false>>(
        Integral<32, false>::from(Val.getZExtValue()));
    break;
  case 16:
    S.Stk.push<Integral<16, false>>(
        Integral<16, false>::from(Val.getZExtValue()));
    break;
  case 8:
    S.Stk.push<Integral<8, false>>(
        Integral<8, false>::from(Val.getZExtValue()));
    break;
  default:
    llvm_unreachable("Invalid integer bitwidth");
  }
}

/// Pushes \p Val to the stack, as a target-dependent 'long'.
static void pushLong(InterpState &S, int64_t Val) {
  PrimType LongType = getLongPrimType(S);
  if (LongType == PT_Sint64)
    S.Stk.push<Integral<64, true>>(Integral<64, true>::from(Val));
  else if (LongType == PT_Sint32)
    S.Stk.push<Integral<32, true>>(Integral<32, true>::from(Val));
  else if (LongType == PT_Sint16)
    S.Stk.push<Integral<16, true>>(Integral<16, true>::from(Val));
  else
    llvm_unreachable("Long isn't 16, 32 or 64 bit?");
}

static void pushSizeT(InterpState &S, uint64_t Val) {
  const TargetInfo &TI = S.getCtx().getTargetInfo();
  unsigned SizeTWidth = TI.getTypeWidth(TI.getSizeType());

  switch (SizeTWidth) {
  case 64:
    S.Stk.push<Integral<64, false>>(Integral<64, false>::from(Val));
    break;
  case 32:
    S.Stk.push<Integral<32, false>>(Integral<32, false>::from(Val));
    break;
  case 16:
    S.Stk.push<Integral<16, false>>(Integral<16, false>::from(Val));
    break;
  default:
    llvm_unreachable("We don't handle this size_t size.");
  }
}

static bool retPrimValue(InterpState &S, CodePtr OpPC, APValue &Result,
                         std::optional<PrimType> &T) {
  if (!T)
    return RetVoid(S, OpPC, Result);

#define RET_CASE(X)                                                            \
  case X:                                                                      \
    return Ret<X>(S, OpPC, Result);
  switch (*T) {
    RET_CASE(PT_Ptr);
    RET_CASE(PT_FnPtr);
    RET_CASE(PT_Float);
    RET_CASE(PT_Bool);
    RET_CASE(PT_Sint8);
    RET_CASE(PT_Uint8);
    RET_CASE(PT_Sint16);
    RET_CASE(PT_Uint16);
    RET_CASE(PT_Sint32);
    RET_CASE(PT_Uint32);
    RET_CASE(PT_Sint64);
    RET_CASE(PT_Uint64);
  default:
    llvm_unreachable("Unsupported return type for builtin function");
  }
#undef RET_CASE
}

static bool interp__builtin_strcmp(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame) {
  const Pointer &A = getParam<Pointer>(Frame, 0);
  const Pointer &B = getParam<Pointer>(Frame, 1);

  if (!CheckLive(S, OpPC, A, AK_Read) || !CheckLive(S, OpPC, B, AK_Read))
    return false;

  assert(A.getFieldDesc()->isPrimitiveArray());
  assert(B.getFieldDesc()->isPrimitiveArray());

  unsigned IndexA = A.getIndex();
  unsigned IndexB = B.getIndex();
  int32_t Result = 0;
  for (;; ++IndexA, ++IndexB) {
    const Pointer &PA = A.atIndex(IndexA);
    const Pointer &PB = B.atIndex(IndexB);
    if (!CheckRange(S, OpPC, PA, AK_Read) ||
        !CheckRange(S, OpPC, PB, AK_Read)) {
      return false;
    }
    uint8_t CA = PA.deref<uint8_t>();
    uint8_t CB = PB.deref<uint8_t>();

    if (CA > CB) {
      Result = 1;
      break;
    } else if (CA < CB) {
      Result = -1;
      break;
    }
    if (CA == 0 || CB == 0)
      break;
  }

  pushInt(S, Result);
  return true;
}

static bool interp__builtin_strlen(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame) {
  const Pointer &StrPtr = getParam<Pointer>(Frame, 0);

  if (!CheckArray(S, OpPC, StrPtr))
    return false;

  if (!CheckLive(S, OpPC, StrPtr, AK_Read))
    return false;

  if (!CheckDummy(S, OpPC, StrPtr))
    return false;

  assert(StrPtr.getFieldDesc()->isPrimitiveArray());

  size_t Len = 0;
  for (size_t I = StrPtr.getIndex();; ++I, ++Len) {
    const Pointer &ElemPtr = StrPtr.atIndex(I);

    if (!CheckRange(S, OpPC, ElemPtr, AK_Read))
      return false;

    uint8_t Val = ElemPtr.deref<uint8_t>();
    if (Val == 0)
      break;
  }

  pushSizeT(S, Len);
  return true;
}

static bool interp__builtin_nan(InterpState &S, CodePtr OpPC,
                                const InterpFrame *Frame, const Function *F,
                                bool Signaling) {
  const Pointer &Arg = getParam<Pointer>(Frame, 0);

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
      S.getCtx().getFloatTypeSemantics(F->getDecl()->getReturnType());

  Floating Result;
  if (S.getCtx().getTargetInfo().isNan2008()) {
    if (Signaling)
      Result = Floating(
          llvm::APFloat::getSNaN(TargetSemantics, /*Negative=*/false, &Fill));
    else
      Result = Floating(
          llvm::APFloat::getQNaN(TargetSemantics, /*Negative=*/false, &Fill));
  } else {
    // Prior to IEEE 754-2008, architectures were allowed to choose whether
    // the first bit of their significand was set for qNaN or sNaN. MIPS chose
    // a different encoding to what became a standard in 2008, and for pre-
    // 2008 revisions, MIPS interpreted sNaN-2008 as qNan and qNaN-2008 as
    // sNaN. This is now known as "legacy NaN" encoding.
    if (Signaling)
      Result = Floating(
          llvm::APFloat::getQNaN(TargetSemantics, /*Negative=*/false, &Fill));
    else
      Result = Floating(
          llvm::APFloat::getSNaN(TargetSemantics, /*Negative=*/false, &Fill));
  }

  S.Stk.push<Floating>(Result);
  return true;
}

static bool interp__builtin_inf(InterpState &S, CodePtr OpPC,
                                const InterpFrame *Frame, const Function *F) {
  const llvm::fltSemantics &TargetSemantics =
      S.getCtx().getFloatTypeSemantics(F->getDecl()->getReturnType());

  S.Stk.push<Floating>(Floating::getInf(TargetSemantics));
  return true;
}

static bool interp__builtin_copysign(InterpState &S, CodePtr OpPC,
                                     const InterpFrame *Frame,
                                     const Function *F) {
  const Floating &Arg1 = getParam<Floating>(Frame, 0);
  const Floating &Arg2 = getParam<Floating>(Frame, 1);

  APFloat Copy = Arg1.getAPFloat();
  Copy.copySign(Arg2.getAPFloat());
  S.Stk.push<Floating>(Floating(Copy));

  return true;
}

static bool interp__builtin_fmin(InterpState &S, CodePtr OpPC,
                                 const InterpFrame *Frame, const Function *F) {
  const Floating &LHS = getParam<Floating>(Frame, 0);
  const Floating &RHS = getParam<Floating>(Frame, 1);

  Floating Result;

  // When comparing zeroes, return -0.0 if one of the zeroes is negative.
  if (LHS.isZero() && RHS.isZero() && RHS.isNegative())
    Result = RHS;
  else if (LHS.isNan() || RHS < LHS)
    Result = RHS;
  else
    Result = LHS;

  S.Stk.push<Floating>(Result);
  return true;
}

static bool interp__builtin_fmax(InterpState &S, CodePtr OpPC,
                                 const InterpFrame *Frame,
                                 const Function *Func) {
  const Floating &LHS = getParam<Floating>(Frame, 0);
  const Floating &RHS = getParam<Floating>(Frame, 1);

  Floating Result;

  // When comparing zeroes, return +0.0 if one of the zeroes is positive.
  if (LHS.isZero() && RHS.isZero() && LHS.isNegative())
    Result = RHS;
  else if (LHS.isNan() || RHS > LHS)
    Result = RHS;
  else
    Result = LHS;

  S.Stk.push<Floating>(Result);
  return true;
}

/// Defined as __builtin_isnan(...), to accommodate the fact that it can
/// take a float, double, long double, etc.
/// But for us, that's all a Floating anyway.
static bool interp__builtin_isnan(InterpState &S, CodePtr OpPC,
                                  const InterpFrame *Frame, const Function *F) {
  const Floating &Arg = S.Stk.peek<Floating>();

  pushInt(S, Arg.isNan());
  return true;
}

static bool interp__builtin_issignaling(InterpState &S, CodePtr OpPC,
                                        const InterpFrame *Frame,
                                        const Function *F) {
  const Floating &Arg = S.Stk.peek<Floating>();

  pushInt(S, Arg.isSignaling());
  return true;
}

static bool interp__builtin_isinf(InterpState &S, CodePtr OpPC,
                                  const InterpFrame *Frame, const Function *F,
                                  bool CheckSign) {
  const Floating &Arg = S.Stk.peek<Floating>();
  bool IsInf = Arg.isInf();

  if (CheckSign)
    pushInt(S, IsInf ? (Arg.isNegative() ? -1 : 1) : 0);
  else
    pushInt(S, Arg.isInf());
  return true;
}

static bool interp__builtin_isfinite(InterpState &S, CodePtr OpPC,
                                     const InterpFrame *Frame,
                                     const Function *F) {
  const Floating &Arg = S.Stk.peek<Floating>();

  pushInt(S, Arg.isFinite());
  return true;
}

static bool interp__builtin_isnormal(InterpState &S, CodePtr OpPC,
                                     const InterpFrame *Frame,
                                     const Function *F) {
  const Floating &Arg = S.Stk.peek<Floating>();

  pushInt(S, Arg.isNormal());
  return true;
}

static bool interp__builtin_issubnormal(InterpState &S, CodePtr OpPC,
                                        const InterpFrame *Frame,
                                        const Function *F) {
  const Floating &Arg = S.Stk.peek<Floating>();

  pushInt(S, Arg.isDenormal());
  return true;
}

static bool interp__builtin_iszero(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const Function *F) {
  const Floating &Arg = S.Stk.peek<Floating>();

  pushInt(S, Arg.isZero());
  return true;
}

/// First parameter to __builtin_isfpclass is the floating value, the
/// second one is an integral value.
static bool interp__builtin_isfpclass(InterpState &S, CodePtr OpPC,
                                      const InterpFrame *Frame,
                                      const Function *Func,
                                      const CallExpr *Call) {
  PrimType FPClassArgT = *S.getContext().classify(Call->getArg(1)->getType());
  APSInt FPClassArg = peekToAPSInt(S.Stk, FPClassArgT);
  const Floating &F =
      S.Stk.peek<Floating>(align(primSize(FPClassArgT) + primSize(PT_Float)));

  int32_t Result =
      static_cast<int32_t>((F.classify() & FPClassArg).getZExtValue());
  pushInt(S, Result);

  return true;
}

/// Five int values followed by one floating value.
static bool interp__builtin_fpclassify(InterpState &S, CodePtr OpPC,
                                       const InterpFrame *Frame,
                                       const Function *Func) {
  const Floating &Val = S.Stk.peek<Floating>();

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
  unsigned IntSize = primSize(getIntPrimType(S));
  unsigned Offset =
      align(primSize(PT_Float)) + ((1 + (4 - Index)) * align(IntSize));

  APSInt I = peekToAPSInt(S.Stk, getIntPrimType(S), Offset);
  pushInt(S, I.getZExtValue());
  return true;
}

// The C standard says "fabs raises no floating-point exceptions,
// even if x is a signaling NaN. The returned value is independent of
// the current rounding direction mode."  Therefore constant folding can
// proceed without regard to the floating point settings.
// Reference, WG14 N2478 F.10.4.3
static bool interp__builtin_fabs(InterpState &S, CodePtr OpPC,
                                 const InterpFrame *Frame,
                                 const Function *Func) {
  const Floating &Val = getParam<Floating>(Frame, 0);

  S.Stk.push<Floating>(Floating::abs(Val));
  return true;
}

static bool interp__builtin_popcount(InterpState &S, CodePtr OpPC,
                                     const InterpFrame *Frame,
                                     const Function *Func,
                                     const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Val = peekToAPSInt(S.Stk, ArgT);
  pushInt(S, Val.popcount());
  return true;
}

static bool interp__builtin_parity(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const Function *Func, const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Val = peekToAPSInt(S.Stk, ArgT);
  pushInt(S, Val.popcount() % 2);
  return true;
}

static bool interp__builtin_clrsb(InterpState &S, CodePtr OpPC,
                                  const InterpFrame *Frame,
                                  const Function *Func, const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Val = peekToAPSInt(S.Stk, ArgT);
  pushInt(S, Val.getBitWidth() - Val.getSignificantBits());
  return true;
}

static bool interp__builtin_bitreverse(InterpState &S, CodePtr OpPC,
                                       const InterpFrame *Frame,
                                       const Function *Func,
                                       const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Val = peekToAPSInt(S.Stk, ArgT);
  pushAPSInt(S, APSInt(Val.reverseBits(), /*IsUnsigned=*/true));
  return true;
}

static bool interp__builtin_classify_type(InterpState &S, CodePtr OpPC,
                                          const InterpFrame *Frame,
                                          const Function *Func,
                                          const CallExpr *Call) {
  // This is an unevaluated call, so there are no arguments on the stack.
  assert(Call->getNumArgs() == 1);
  const Expr *Arg = Call->getArg(0);

  GCCTypeClass ResultClass =
      EvaluateBuiltinClassifyType(Arg->getType(), S.getLangOpts());
  int32_t ReturnVal = static_cast<int32_t>(ResultClass);
  pushInt(S, ReturnVal);
  return true;
}

// __builtin_expect(long, long)
// __builtin_expect_with_probability(long, long, double)
static bool interp__builtin_expect(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const Function *Func, const CallExpr *Call) {
  // The return value is simply the value of the first parameter.
  // We ignore the probability.
  unsigned NumArgs = Call->getNumArgs();
  assert(NumArgs == 2 || NumArgs == 3);

  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  unsigned Offset = align(primSize(getLongPrimType(S))) * 2;
  if (NumArgs == 3)
    Offset += align(primSize(PT_Float));

  APSInt Val = peekToAPSInt(S.Stk, ArgT, Offset);
  pushLong(S, Val.getSExtValue());
  return true;
}

/// rotateleft(value, amount)
static bool interp__builtin_rotate(InterpState &S, CodePtr OpPC,
                                   const InterpFrame *Frame,
                                   const Function *Func, const CallExpr *Call,
                                   bool Right) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  assert(ArgT == *S.getContext().classify(Call->getArg(1)->getType()));

  APSInt Amount = peekToAPSInt(S.Stk, ArgT);
  APSInt Value = peekToAPSInt(S.Stk, ArgT, align(primSize(ArgT)) * 2);

  APSInt Result;
  if (Right)
    Result = APSInt(Value.rotr(Amount.urem(Value.getBitWidth())),
                    /*IsUnsigned=*/true);
  else // Left.
    Result = APSInt(Value.rotl(Amount.urem(Value.getBitWidth())),
                    /*IsUnsigned=*/true);

  pushAPSInt(S, Result);
  return true;
}

static bool interp__builtin_ffs(InterpState &S, CodePtr OpPC,
                                const InterpFrame *Frame, const Function *Func,
                                const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Value = peekToAPSInt(S.Stk, ArgT);

  uint64_t N = Value.countr_zero();
  pushInt(S, N == Value.getBitWidth() ? 0 : N + 1);
  return true;
}

static bool interp__builtin_addressof(InterpState &S, CodePtr OpPC,
                                      const InterpFrame *Frame,
                                      const Function *Func,
                                      const CallExpr *Call) {
  PrimType PtrT =
      S.getContext().classify(Call->getArg(0)->getType()).value_or(PT_Ptr);

  if (PtrT == PT_FnPtr) {
    const FunctionPointer &Arg = S.Stk.peek<FunctionPointer>();
    S.Stk.push<FunctionPointer>(Arg);
  } else if (PtrT == PT_Ptr) {
    const Pointer &Arg = S.Stk.peek<Pointer>();
    S.Stk.push<Pointer>(Arg);
  } else {
    assert(false && "Unsupported pointer type passed to __builtin_addressof()");
  }
  return true;
}

static bool interp__builtin_move(InterpState &S, CodePtr OpPC,
                                 const InterpFrame *Frame, const Function *Func,
                                 const CallExpr *Call) {

  PrimType ArgT = S.getContext().classify(Call->getArg(0)).value_or(PT_Ptr);

  TYPE_SWITCH(ArgT, const T &Arg = S.Stk.peek<T>(); S.Stk.push<T>(Arg););

  return Func->getDecl()->isConstexpr();
}

static bool interp__builtin_eh_return_data_regno(InterpState &S, CodePtr OpPC,
                                                 const InterpFrame *Frame,
                                                 const Function *Func,
                                                 const CallExpr *Call) {
  PrimType ArgT = *S.getContext().classify(Call->getArg(0)->getType());
  APSInt Arg = peekToAPSInt(S.Stk, ArgT);

  int Result =
      S.getCtx().getTargetInfo().getEHDataRegisterNumber(Arg.getZExtValue());
  pushInt(S, Result);
  return true;
}

bool InterpretBuiltin(InterpState &S, CodePtr OpPC, const Function *F,
                      const CallExpr *Call) {
  InterpFrame *Frame = S.Current;
  APValue Dummy;

  std::optional<PrimType> ReturnT = S.getContext().classify(Call);

  // If classify failed, we assume void.
  assert(ReturnT || Call->getType()->isVoidType());

  switch (F->getBuiltinID()) {
  case Builtin::BI__builtin_is_constant_evaluated:
    S.Stk.push<Boolean>(Boolean::from(S.inConstantContext()));
    break;
  case Builtin::BI__builtin_assume:
  case Builtin::BI__assume:
    break;
  case Builtin::BI__builtin_strcmp:
    if (!interp__builtin_strcmp(S, OpPC, Frame))
      return false;
    break;
  case Builtin::BI__builtin_strlen:
    if (!interp__builtin_strlen(S, OpPC, Frame))
      return false;
    break;
  case Builtin::BI__builtin_nan:
  case Builtin::BI__builtin_nanf:
  case Builtin::BI__builtin_nanl:
  case Builtin::BI__builtin_nanf16:
  case Builtin::BI__builtin_nanf128:
    if (!interp__builtin_nan(S, OpPC, Frame, F, /*Signaling=*/false))
      return false;
    break;
  case Builtin::BI__builtin_nans:
  case Builtin::BI__builtin_nansf:
  case Builtin::BI__builtin_nansl:
  case Builtin::BI__builtin_nansf16:
  case Builtin::BI__builtin_nansf128:
    if (!interp__builtin_nan(S, OpPC, Frame, F, /*Signaling=*/true))
      return false;
    break;

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
    if (!interp__builtin_inf(S, OpPC, Frame, F))
      return false;
    break;
  case Builtin::BI__builtin_copysign:
  case Builtin::BI__builtin_copysignf:
  case Builtin::BI__builtin_copysignl:
  case Builtin::BI__builtin_copysignf128:
    if (!interp__builtin_copysign(S, OpPC, Frame, F))
      return false;
    break;

  case Builtin::BI__builtin_fmin:
  case Builtin::BI__builtin_fminf:
  case Builtin::BI__builtin_fminl:
  case Builtin::BI__builtin_fminf16:
  case Builtin::BI__builtin_fminf128:
    if (!interp__builtin_fmin(S, OpPC, Frame, F))
      return false;
    break;

  case Builtin::BI__builtin_fmax:
  case Builtin::BI__builtin_fmaxf:
  case Builtin::BI__builtin_fmaxl:
  case Builtin::BI__builtin_fmaxf16:
  case Builtin::BI__builtin_fmaxf128:
    if (!interp__builtin_fmax(S, OpPC, Frame, F))
      return false;
    break;

  case Builtin::BI__builtin_isnan:
    if (!interp__builtin_isnan(S, OpPC, Frame, F))
      return false;
    break;
  case Builtin::BI__builtin_issignaling:
    if (!interp__builtin_issignaling(S, OpPC, Frame, F))
      return false;
    break;

  case Builtin::BI__builtin_isinf:
    if (!interp__builtin_isinf(S, OpPC, Frame, F, /*Sign=*/false))
      return false;
    break;

  case Builtin::BI__builtin_isinf_sign:
    if (!interp__builtin_isinf(S, OpPC, Frame, F, /*Sign=*/true))
      return false;
    break;

  case Builtin::BI__builtin_isfinite:
    if (!interp__builtin_isfinite(S, OpPC, Frame, F))
      return false;
    break;
  case Builtin::BI__builtin_isnormal:
    if (!interp__builtin_isnormal(S, OpPC, Frame, F))
      return false;
    break;
  case Builtin::BI__builtin_issubnormal:
    if (!interp__builtin_issubnormal(S, OpPC, Frame, F))
      return false;
    break;
  case Builtin::BI__builtin_iszero:
    if (!interp__builtin_iszero(S, OpPC, Frame, F))
      return false;
    break;
  case Builtin::BI__builtin_isfpclass:
    if (!interp__builtin_isfpclass(S, OpPC, Frame, F, Call))
      return false;
    break;
  case Builtin::BI__builtin_fpclassify:
    if (!interp__builtin_fpclassify(S, OpPC, Frame, F))
      return false;
    break;

  case Builtin::BI__builtin_fabs:
  case Builtin::BI__builtin_fabsf:
  case Builtin::BI__builtin_fabsl:
  case Builtin::BI__builtin_fabsf128:
    if (!interp__builtin_fabs(S, OpPC, Frame, F))
      return false;
    break;

  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll:
  case Builtin::BI__popcnt16: // Microsoft variants of popcount
  case Builtin::BI__popcnt:
  case Builtin::BI__popcnt64:
    if (!interp__builtin_popcount(S, OpPC, Frame, F, Call))
      return false;
    break;

  case Builtin::BI__builtin_parity:
  case Builtin::BI__builtin_parityl:
  case Builtin::BI__builtin_parityll:
    if (!interp__builtin_parity(S, OpPC, Frame, F, Call))
      return false;
    break;

  case Builtin::BI__builtin_clrsb:
  case Builtin::BI__builtin_clrsbl:
  case Builtin::BI__builtin_clrsbll:
    if (!interp__builtin_clrsb(S, OpPC, Frame, F, Call))
      return false;
    break;

  case Builtin::BI__builtin_bitreverse8:
  case Builtin::BI__builtin_bitreverse16:
  case Builtin::BI__builtin_bitreverse32:
  case Builtin::BI__builtin_bitreverse64:
    if (!interp__builtin_bitreverse(S, OpPC, Frame, F, Call))
      return false;
    break;

  case Builtin::BI__builtin_classify_type:
    if (!interp__builtin_classify_type(S, OpPC, Frame, F, Call))
      return false;
    break;

  case Builtin::BI__builtin_expect:
  case Builtin::BI__builtin_expect_with_probability:
    if (!interp__builtin_expect(S, OpPC, Frame, F, Call))
      return false;
    break;

  case Builtin::BI__builtin_rotateleft8:
  case Builtin::BI__builtin_rotateleft16:
  case Builtin::BI__builtin_rotateleft32:
  case Builtin::BI__builtin_rotateleft64:
  case Builtin::BI_rotl8: // Microsoft variants of rotate left
  case Builtin::BI_rotl16:
  case Builtin::BI_rotl:
  case Builtin::BI_lrotl:
  case Builtin::BI_rotl64:
    if (!interp__builtin_rotate(S, OpPC, Frame, F, Call, /*Right=*/false))
      return false;
    break;

  case Builtin::BI__builtin_rotateright8:
  case Builtin::BI__builtin_rotateright16:
  case Builtin::BI__builtin_rotateright32:
  case Builtin::BI__builtin_rotateright64:
  case Builtin::BI_rotr8: // Microsoft variants of rotate right
  case Builtin::BI_rotr16:
  case Builtin::BI_rotr:
  case Builtin::BI_lrotr:
  case Builtin::BI_rotr64:
    if (!interp__builtin_rotate(S, OpPC, Frame, F, Call, /*Right=*/true))
      return false;
    break;

  case Builtin::BI__builtin_ffs:
  case Builtin::BI__builtin_ffsl:
  case Builtin::BI__builtin_ffsll:
    if (!interp__builtin_ffs(S, OpPC, Frame, F, Call))
      return false;
    break;
  case Builtin::BIaddressof:
  case Builtin::BI__addressof:
  case Builtin::BI__builtin_addressof:
    if (!interp__builtin_addressof(S, OpPC, Frame, F, Call))
      return false;
    break;

  case Builtin::BIas_const:
  case Builtin::BIforward:
  case Builtin::BIforward_like:
  case Builtin::BImove:
  case Builtin::BImove_if_noexcept:
    if (!interp__builtin_move(S, OpPC, Frame, F, Call))
      return false;
    break;

  case Builtin::BI__builtin_eh_return_data_regno:
    if (!interp__builtin_eh_return_data_regno(S, OpPC, Frame, F, Call))
      return false;
    break;

  default:
    return false;
  }

  return retPrimValue(S, OpPC, Dummy, ReturnT);
}

bool InterpretOffsetOf(InterpState &S, CodePtr OpPC, const OffsetOfExpr *E,
                       llvm::ArrayRef<int64_t> ArrayIndices,
                       int64_t &IntResult) {
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
      const RecordType *RT = CurrentType->getAs<RecordType>();
      if (!RT)
        return false;
      RecordDecl *RD = RT->getDecl();
      if (RD->isInvalidDecl())
        return false;
      const ASTRecordLayout &RL = S.getCtx().getASTRecordLayout(RD);
      unsigned FieldIndex = MemberDecl->getFieldIndex();
      assert(FieldIndex < RL.getFieldCount() && "offsetof field in wrong type");
      Result += S.getCtx().toCharUnitsFromBits(RL.getFieldOffset(FieldIndex));
      CurrentType = MemberDecl->getType().getNonReferenceType();
      break;
    }
    case OffsetOfNode::Array: {
      // When generating bytecode, we put all the index expressions as Sint64 on
      // the stack.
      int64_t Index = ArrayIndices[ArrayIndex];
      const ArrayType *AT = S.getCtx().getAsArrayType(CurrentType);
      if (!AT)
        return false;
      CurrentType = AT->getElementType();
      CharUnits ElementSize = S.getCtx().getTypeSizeInChars(CurrentType);
      Result += Index * ElementSize;
      ++ArrayIndex;
      break;
    }
    case OffsetOfNode::Base: {
      const CXXBaseSpecifier *BaseSpec = Node.getBase();
      if (BaseSpec->isVirtual())
        return false;

      // Find the layout of the class whose base we are looking into.
      const RecordType *RT = CurrentType->getAs<RecordType>();
      if (!RT)
        return false;
      const RecordDecl *RD = RT->getDecl();
      if (RD->isInvalidDecl())
        return false;
      const ASTRecordLayout &RL = S.getCtx().getASTRecordLayout(RD);

      // Find the base class itself.
      CurrentType = BaseSpec->getType();
      const RecordType *BaseRT = CurrentType->getAs<RecordType>();
      if (!BaseRT)
        return false;

      // Add the offset to the base.
      Result += RL.getBaseClassOffset(cast<CXXRecordDecl>(BaseRT->getDecl()));
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

} // namespace interp
} // namespace clang
