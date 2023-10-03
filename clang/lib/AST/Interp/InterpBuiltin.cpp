//===--- InterpBuiltin.cpp - Interpreter for the constexpr VM ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

/// Peek an integer value from the stack into an APSInt.
static APSInt peekToAPSInt(InterpStack &Stk, PrimType T, size_t Offset = 0) {
  if (Offset == 0)
    Offset = align(primSize(T));

  APSInt R;
  INT_TYPE_SWITCH(T, {
    T Val = Stk.peek<T>(Offset);
    R = APSInt(APInt(T::bitWidth(), static_cast<uint64_t>(Val), T::isSigned()));
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

static bool retInt(InterpState &S, CodePtr OpPC, APValue &Result) {
  PrimType IntType = getIntPrimType(S);
  if (IntType == PT_Sint32)
    return Ret<PT_Sint32>(S, OpPC, Result);
  else if (IntType == PT_Sint16)
    return Ret<PT_Sint16>(S, OpPC, Result);
  llvm_unreachable("Int isn't 16 or 32 bit?");
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

static bool retSizeT(InterpState &S, CodePtr OpPC, APValue &Result) {
  const TargetInfo &TI = S.getCtx().getTargetInfo();
  unsigned SizeTWidth = TI.getTypeWidth(TI.getSizeType());

  switch (SizeTWidth) {
  case 64:
    return Ret<PT_Uint64>(S, OpPC, Result);
  case 32:
    return Ret<PT_Uint32>(S, OpPC, Result);
  case 16:
    return Ret<PT_Uint16>(S, OpPC, Result);
  }

  llvm_unreachable("size_t isn't 64 or 32 bit?");
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

bool InterpretBuiltin(InterpState &S, CodePtr OpPC, const Function *F,
                      const CallExpr *Call) {
  InterpFrame *Frame = S.Current;
  APValue Dummy;

  switch (F->getBuiltinID()) {
  case Builtin::BI__builtin_is_constant_evaluated:
    S.Stk.push<Boolean>(Boolean::from(S.inConstantContext()));
    return Ret<PT_Bool>(S, OpPC, Dummy);
  case Builtin::BI__builtin_assume:
    return RetVoid(S, OpPC, Dummy);
  case Builtin::BI__builtin_strcmp:
    if (interp__builtin_strcmp(S, OpPC, Frame))
      return retInt(S, OpPC, Dummy);
    break;
  case Builtin::BI__builtin_strlen:
    if (interp__builtin_strlen(S, OpPC, Frame))
      return retSizeT(S, OpPC, Dummy);
    break;
  case Builtin::BI__builtin_nan:
  case Builtin::BI__builtin_nanf:
  case Builtin::BI__builtin_nanl:
  case Builtin::BI__builtin_nanf16:
  case Builtin::BI__builtin_nanf128:
    if (interp__builtin_nan(S, OpPC, Frame, F, /*Signaling=*/false))
      return Ret<PT_Float>(S, OpPC, Dummy);
    break;
  case Builtin::BI__builtin_nans:
  case Builtin::BI__builtin_nansf:
  case Builtin::BI__builtin_nansl:
  case Builtin::BI__builtin_nansf16:
  case Builtin::BI__builtin_nansf128:
    if (interp__builtin_nan(S, OpPC, Frame, F, /*Signaling=*/true))
      return Ret<PT_Float>(S, OpPC, Dummy);
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
    if (interp__builtin_inf(S, OpPC, Frame, F))
      return Ret<PT_Float>(S, OpPC, Dummy);
    break;
  case Builtin::BI__builtin_copysign:
  case Builtin::BI__builtin_copysignf:
  case Builtin::BI__builtin_copysignl:
  case Builtin::BI__builtin_copysignf128:
    if (interp__builtin_copysign(S, OpPC, Frame, F))
      return Ret<PT_Float>(S, OpPC, Dummy);
    break;

  case Builtin::BI__builtin_fmin:
  case Builtin::BI__builtin_fminf:
  case Builtin::BI__builtin_fminl:
  case Builtin::BI__builtin_fminf16:
  case Builtin::BI__builtin_fminf128:
    if (interp__builtin_fmin(S, OpPC, Frame, F))
      return Ret<PT_Float>(S, OpPC, Dummy);
    break;

  case Builtin::BI__builtin_fmax:
  case Builtin::BI__builtin_fmaxf:
  case Builtin::BI__builtin_fmaxl:
  case Builtin::BI__builtin_fmaxf16:
  case Builtin::BI__builtin_fmaxf128:
    if (interp__builtin_fmax(S, OpPC, Frame, F))
      return Ret<PT_Float>(S, OpPC, Dummy);
    break;

  case Builtin::BI__builtin_isnan:
    if (interp__builtin_isnan(S, OpPC, Frame, F))
      return retInt(S, OpPC, Dummy);
    break;

  case Builtin::BI__builtin_isinf:
    if (interp__builtin_isinf(S, OpPC, Frame, F, /*Sign=*/false))
      return retInt(S, OpPC, Dummy);
    break;

  case Builtin::BI__builtin_isinf_sign:
    if (interp__builtin_isinf(S, OpPC, Frame, F, /*Sign=*/true))
      return retInt(S, OpPC, Dummy);
    break;

  case Builtin::BI__builtin_isfinite:
    if (interp__builtin_isfinite(S, OpPC, Frame, F))
      return retInt(S, OpPC, Dummy);
    break;
  case Builtin::BI__builtin_isnormal:
    if (interp__builtin_isnormal(S, OpPC, Frame, F))
      return retInt(S, OpPC, Dummy);
    break;
  case Builtin::BI__builtin_isfpclass:
    if (interp__builtin_isfpclass(S, OpPC, Frame, F, Call))
      return retInt(S, OpPC, Dummy);
    break;
  case Builtin::BI__builtin_fpclassify:
    if (interp__builtin_fpclassify(S, OpPC, Frame, F))
      return retInt(S, OpPC, Dummy);
    break;

  case Builtin::BI__builtin_fabs:
  case Builtin::BI__builtin_fabsf:
  case Builtin::BI__builtin_fabsl:
  case Builtin::BI__builtin_fabsf128:
    if (interp__builtin_fabs(S, OpPC, Frame, F))
      return Ret<PT_Float>(S, OpPC, Dummy);
    break;

  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll:
  case Builtin::BI__popcnt16: // Microsoft variants of popcount
  case Builtin::BI__popcnt:
  case Builtin::BI__popcnt64:
    if (interp__builtin_popcount(S, OpPC, Frame, F, Call))
      return retInt(S, OpPC, Dummy);
    break;

  default:
    return false;
  }

  return false;
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
