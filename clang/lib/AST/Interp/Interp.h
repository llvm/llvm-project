//===--- Interp.h - Interpreter for the constexpr VM ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of the interpreter state and entry point.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_INTERP_H
#define LLVM_CLANG_AST_INTERP_INTERP_H

#include "Boolean.h"
#include "Floating.h"
#include "Function.h"
#include "FunctionPointer.h"
#include "InterpFrame.h"
#include "InterpStack.h"
#include "InterpState.h"
#include "Opcode.h"
#include "PrimType.h"
#include "Program.h"
#include "State.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Endian.h"
#include <limits>
#include <type_traits>

namespace clang {
namespace interp {

using APSInt = llvm::APSInt;

/// Convert a value to an APValue.
template <typename T> bool ReturnValue(const T &V, APValue &R) {
  R = V.toAPValue();
  return true;
}

/// Checks if the variable has externally defined storage.
bool CheckExtern(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if the array is offsetable.
bool CheckArray(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a pointer is live and accessible.
bool CheckLive(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               AccessKinds AK);

/// Checks if a pointer is a dummy pointer.
bool CheckDummy(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a pointer is null.
bool CheckNull(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               CheckSubobjectKind CSK);

/// Checks if a pointer is in range.
bool CheckRange(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                AccessKinds AK);

/// Checks if a field from which a pointer is going to be derived is valid.
bool CheckRange(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                CheckSubobjectKind CSK);

/// Checks if Ptr is a one-past-the-end pointer.
bool CheckSubobject(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                    CheckSubobjectKind CSK);

/// Checks if a pointer points to const storage.
bool CheckConst(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if the Descriptor is of a constexpr or const global variable.
bool CheckConstant(InterpState &S, CodePtr OpPC, const Descriptor *Desc);

/// Checks if a pointer points to a mutable field.
bool CheckMutable(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a value can be loaded from a block.
bool CheckLoad(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

bool CheckInitialized(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                      AccessKinds AK);
/// Check if a global variable is initialized.
bool CheckGlobalInitialized(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a value can be stored in a block.
bool CheckStore(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a method can be invoked on an object.
bool CheckInvoke(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a value can be initialized.
bool CheckInit(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a method can be called.
bool CheckCallable(InterpState &S, CodePtr OpPC, const Function *F);

/// Checks if calling the currently active function would exceed
/// the allowed call depth.
bool CheckCallDepth(InterpState &S, CodePtr OpPC);

/// Checks the 'this' pointer.
bool CheckThis(InterpState &S, CodePtr OpPC, const Pointer &This);

/// Checks if a method is pure virtual.
bool CheckPure(InterpState &S, CodePtr OpPC, const CXXMethodDecl *MD);

/// Checks if reinterpret casts are legal in the current context.
bool CheckPotentialReinterpretCast(InterpState &S, CodePtr OpPC,
                                   const Pointer &Ptr);

/// Sets the given integral value to the pointer, which is of
/// a std::{weak,partial,strong}_ordering type.
bool SetThreeWayComparisonField(InterpState &S, CodePtr OpPC,
                                const Pointer &Ptr, const APSInt &IntValue);

/// Checks if the shift operation is legal.
template <typename LT, typename RT>
bool CheckShift(InterpState &S, CodePtr OpPC, const LT &LHS, const RT &RHS,
                unsigned Bits) {
  if (RHS.isNegative()) {
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.CCEDiag(Loc, diag::note_constexpr_negative_shift) << RHS.toAPSInt();
    return false;
  }

  // C++11 [expr.shift]p1: Shift width must be less than the bit width of
  // the shifted type.
  if (Bits > 1 && RHS >= RT::from(Bits, RHS.bitWidth())) {
    const Expr *E = S.Current->getExpr(OpPC);
    const APSInt Val = RHS.toAPSInt();
    QualType Ty = E->getType();
    S.CCEDiag(E, diag::note_constexpr_large_shift) << Val << Ty << Bits;
    return false;
  }

  if (LHS.isSigned() && !S.getLangOpts().CPlusPlus20) {
    const Expr *E = S.Current->getExpr(OpPC);
    // C++11 [expr.shift]p2: A signed left shift must have a non-negative
    // operand, and must not overflow the corresponding unsigned type.
    if (LHS.isNegative())
      S.CCEDiag(E, diag::note_constexpr_lshift_of_negative) << LHS.toAPSInt();
    else if (LHS.toUnsigned().countLeadingZeros() < static_cast<unsigned>(RHS))
      S.CCEDiag(E, diag::note_constexpr_lshift_discards);
  }

  // C++2a [expr.shift]p2: [P0907R4]:
  //    E1 << E2 is the unique value congruent to
  //    E1 x 2^E2 module 2^N.
  return true;
}

/// Checks if Div/Rem operation on LHS and RHS is valid.
template <typename T>
bool CheckDivRem(InterpState &S, CodePtr OpPC, const T &LHS, const T &RHS) {
  if (RHS.isZero()) {
    const auto *Op = cast<BinaryOperator>(S.Current->getExpr(OpPC));
    S.FFDiag(Op, diag::note_expr_divide_by_zero)
        << Op->getRHS()->getSourceRange();
    return false;
  }

  if (LHS.isSigned() && LHS.isMin() && RHS.isNegative() && RHS.isMinusOne()) {
    APSInt LHSInt = LHS.toAPSInt();
    SmallString<32> Trunc;
    (-LHSInt.extend(LHSInt.getBitWidth() + 1)).toString(Trunc, 10);
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    const Expr *E = S.Current->getExpr(OpPC);
    S.CCEDiag(Loc, diag::note_constexpr_overflow) << Trunc << E->getType();
    return false;
  }
  return true;
}

/// Checks if the result of a floating-point operation is valid
/// in the current context.
bool CheckFloatResult(InterpState &S, CodePtr OpPC, const Floating &Result,
                      APFloat::opStatus Status);

/// Checks why the given DeclRefExpr is invalid.
bool CheckDeclRef(InterpState &S, CodePtr OpPC, const DeclRefExpr *DR);

/// Interpreter entry point.
bool Interpret(InterpState &S, APValue &Result);

/// Interpret a builtin function.
bool InterpretBuiltin(InterpState &S, CodePtr OpPC, const Function *F,
                      const CallExpr *Call);

/// Interpret an offsetof operation.
bool InterpretOffsetOf(InterpState &S, CodePtr OpPC, const OffsetOfExpr *E,
                       llvm::ArrayRef<int64_t> ArrayIndices, int64_t &Result);

enum class ArithOp { Add, Sub };

//===----------------------------------------------------------------------===//
// Returning values
//===----------------------------------------------------------------------===//

void cleanupAfterFunctionCall(InterpState &S, CodePtr OpPC);

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Ret(InterpState &S, CodePtr &PC, APValue &Result) {
  const T &Ret = S.Stk.pop<T>();

  // Make sure returned pointers are live. We might be trying to return a
  // pointer or reference to a local variable.
  // Just return false, since a diagnostic has already been emitted in Sema.
  if constexpr (std::is_same_v<T, Pointer>) {
    // FIXME: We could be calling isLive() here, but the emitted diagnostics
    // seem a little weird, at least if the returned expression is of
    // pointer type.
    // Null pointers are considered live here.
    if (!Ret.isZero() && !Ret.isLive())
      return false;
  }

  assert(S.Current);
  assert(S.Current->getFrameOffset() == S.Stk.size() && "Invalid frame");
  if (!S.checkingPotentialConstantExpression() || S.Current->Caller)
    cleanupAfterFunctionCall(S, PC);

  if (InterpFrame *Caller = S.Current->Caller) {
    PC = S.Current->getRetPC();
    delete S.Current;
    S.Current = Caller;
    S.Stk.push<T>(Ret);
  } else {
    delete S.Current;
    S.Current = nullptr;
    if (!ReturnValue<T>(Ret, Result))
      return false;
  }
  return true;
}

inline bool RetVoid(InterpState &S, CodePtr &PC, APValue &Result) {
  assert(S.Current->getFrameOffset() == S.Stk.size() && "Invalid frame");

  if (!S.checkingPotentialConstantExpression() || S.Current->Caller)
    cleanupAfterFunctionCall(S, PC);

  if (InterpFrame *Caller = S.Current->Caller) {
    PC = S.Current->getRetPC();
    delete S.Current;
    S.Current = Caller;
  } else {
    delete S.Current;
    S.Current = nullptr;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Add, Sub, Mul
//===----------------------------------------------------------------------===//

template <typename T, bool (*OpFW)(T, T, unsigned, T *),
          template <typename U> class OpAP>
bool AddSubMulHelper(InterpState &S, CodePtr OpPC, unsigned Bits, const T &LHS,
                     const T &RHS) {
  // Fast path - add the numbers with fixed width.
  T Result;
  if (!OpFW(LHS, RHS, Bits, &Result)) {
    S.Stk.push<T>(Result);
    return true;
  }

  // If for some reason evaluation continues, use the truncated results.
  S.Stk.push<T>(Result);

  // Slow path - compute the result using another bit of precision.
  APSInt Value = OpAP<APSInt>()(LHS.toAPSInt(Bits), RHS.toAPSInt(Bits));

  // Report undefined behaviour, stopping if required.
  const Expr *E = S.Current->getExpr(OpPC);
  QualType Type = E->getType();
  if (S.checkingForUndefinedBehavior()) {
    SmallString<32> Trunc;
    Value.trunc(Result.bitWidth()).toString(Trunc, 10);
    auto Loc = E->getExprLoc();
    S.report(Loc, diag::warn_integer_constant_overflow)
        << Trunc << Type << E->getSourceRange();
    return true;
  } else {
    S.CCEDiag(E, diag::note_constexpr_overflow) << Value << Type;
    if (!S.noteUndefinedBehavior()) {
      S.Stk.pop<T>();
      return false;
    }
    return true;
  }
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Add(InterpState &S, CodePtr OpPC) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();
  const unsigned Bits = RHS.bitWidth() + 1;
  return AddSubMulHelper<T, T::add, std::plus>(S, OpPC, Bits, LHS, RHS);
}

inline bool Addf(InterpState &S, CodePtr OpPC, llvm::RoundingMode RM) {
  const Floating &RHS = S.Stk.pop<Floating>();
  const Floating &LHS = S.Stk.pop<Floating>();

  Floating Result;
  auto Status = Floating::add(LHS, RHS, RM, &Result);
  S.Stk.push<Floating>(Result);
  return CheckFloatResult(S, OpPC, Result, Status);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Sub(InterpState &S, CodePtr OpPC) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();
  const unsigned Bits = RHS.bitWidth() + 1;
  return AddSubMulHelper<T, T::sub, std::minus>(S, OpPC, Bits, LHS, RHS);
}

inline bool Subf(InterpState &S, CodePtr OpPC, llvm::RoundingMode RM) {
  const Floating &RHS = S.Stk.pop<Floating>();
  const Floating &LHS = S.Stk.pop<Floating>();

  Floating Result;
  auto Status = Floating::sub(LHS, RHS, RM, &Result);
  S.Stk.push<Floating>(Result);
  return CheckFloatResult(S, OpPC, Result, Status);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Mul(InterpState &S, CodePtr OpPC) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();
  const unsigned Bits = RHS.bitWidth() * 2;
  return AddSubMulHelper<T, T::mul, std::multiplies>(S, OpPC, Bits, LHS, RHS);
}

inline bool Mulf(InterpState &S, CodePtr OpPC, llvm::RoundingMode RM) {
  const Floating &RHS = S.Stk.pop<Floating>();
  const Floating &LHS = S.Stk.pop<Floating>();

  Floating Result;
  auto Status = Floating::mul(LHS, RHS, RM, &Result);
  S.Stk.push<Floating>(Result);
  return CheckFloatResult(S, OpPC, Result, Status);
}
/// 1) Pops the RHS from the stack.
/// 2) Pops the LHS from the stack.
/// 3) Pushes 'LHS & RHS' on the stack
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool BitAnd(InterpState &S, CodePtr OpPC) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();

  unsigned Bits = RHS.bitWidth();
  T Result;
  if (!T::bitAnd(LHS, RHS, Bits, &Result)) {
    S.Stk.push<T>(Result);
    return true;
  }
  return false;
}

/// 1) Pops the RHS from the stack.
/// 2) Pops the LHS from the stack.
/// 3) Pushes 'LHS | RHS' on the stack
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool BitOr(InterpState &S, CodePtr OpPC) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();

  unsigned Bits = RHS.bitWidth();
  T Result;
  if (!T::bitOr(LHS, RHS, Bits, &Result)) {
    S.Stk.push<T>(Result);
    return true;
  }
  return false;
}

/// 1) Pops the RHS from the stack.
/// 2) Pops the LHS from the stack.
/// 3) Pushes 'LHS ^ RHS' on the stack
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool BitXor(InterpState &S, CodePtr OpPC) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();

  unsigned Bits = RHS.bitWidth();
  T Result;
  if (!T::bitXor(LHS, RHS, Bits, &Result)) {
    S.Stk.push<T>(Result);
    return true;
  }
  return false;
}

/// 1) Pops the RHS from the stack.
/// 2) Pops the LHS from the stack.
/// 3) Pushes 'LHS % RHS' on the stack (the remainder of dividing LHS by RHS).
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Rem(InterpState &S, CodePtr OpPC) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();

  if (!CheckDivRem(S, OpPC, LHS, RHS))
    return false;

  const unsigned Bits = RHS.bitWidth() * 2;
  T Result;
  if (!T::rem(LHS, RHS, Bits, &Result)) {
    S.Stk.push<T>(Result);
    return true;
  }
  return false;
}

/// 1) Pops the RHS from the stack.
/// 2) Pops the LHS from the stack.
/// 3) Pushes 'LHS / RHS' on the stack
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Div(InterpState &S, CodePtr OpPC) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();

  if (!CheckDivRem(S, OpPC, LHS, RHS))
    return false;

  const unsigned Bits = RHS.bitWidth() * 2;
  T Result;
  if (!T::div(LHS, RHS, Bits, &Result)) {
    S.Stk.push<T>(Result);
    return true;
  }
  return false;
}

inline bool Divf(InterpState &S, CodePtr OpPC, llvm::RoundingMode RM) {
  const Floating &RHS = S.Stk.pop<Floating>();
  const Floating &LHS = S.Stk.pop<Floating>();

  if (!CheckDivRem(S, OpPC, LHS, RHS))
    return false;

  Floating Result;
  auto Status = Floating::div(LHS, RHS, RM, &Result);
  S.Stk.push<Floating>(Result);
  return CheckFloatResult(S, OpPC, Result, Status);
}

//===----------------------------------------------------------------------===//
// Inv
//===----------------------------------------------------------------------===//

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Inv(InterpState &S, CodePtr OpPC) {
  using BoolT = PrimConv<PT_Bool>::T;
  const T &Val = S.Stk.pop<T>();
  const unsigned Bits = Val.bitWidth();
  Boolean R;
  Boolean::inv(BoolT::from(Val, Bits), &R);

  S.Stk.push<BoolT>(R);
  return true;
}

//===----------------------------------------------------------------------===//
// Neg
//===----------------------------------------------------------------------===//

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Neg(InterpState &S, CodePtr OpPC) {
  const T &Value = S.Stk.pop<T>();
  T Result;

  if (!T::neg(Value, &Result)) {
    S.Stk.push<T>(Result);
    return true;
  }

  assert(isIntegralType(Name) &&
         "don't expect other types to fail at constexpr negation");
  S.Stk.push<T>(Result);

  APSInt NegatedValue = -Value.toAPSInt(Value.bitWidth() + 1);
  const Expr *E = S.Current->getExpr(OpPC);
  QualType Type = E->getType();

  if (S.checkingForUndefinedBehavior()) {
    SmallString<32> Trunc;
    NegatedValue.trunc(Result.bitWidth()).toString(Trunc, 10);
    auto Loc = E->getExprLoc();
    S.report(Loc, diag::warn_integer_constant_overflow)
        << Trunc << Type << E->getSourceRange();
    return true;
  }

  S.CCEDiag(E, diag::note_constexpr_overflow) << NegatedValue << Type;
  return S.noteUndefinedBehavior();
}

enum class PushVal : bool {
  No,
  Yes,
};
enum class IncDecOp {
  Inc,
  Dec,
};

template <typename T, IncDecOp Op, PushVal DoPush>
bool IncDecHelper(InterpState &S, CodePtr OpPC, const Pointer &Ptr) {
  if (Ptr.isDummy())
    return false;

  const T &Value = Ptr.deref<T>();
  T Result;

  if constexpr (DoPush == PushVal::Yes)
    S.Stk.push<T>(Value);

  if constexpr (Op == IncDecOp::Inc) {
    if (!T::increment(Value, &Result)) {
      Ptr.deref<T>() = Result;
      return true;
    }
  } else {
    if (!T::decrement(Value, &Result)) {
      Ptr.deref<T>() = Result;
      return true;
    }
  }

  // Something went wrong with the previous operation. Compute the
  // result with another bit of precision.
  unsigned Bits = Value.bitWidth() + 1;
  APSInt APResult;
  if constexpr (Op == IncDecOp::Inc)
    APResult = ++Value.toAPSInt(Bits);
  else
    APResult = --Value.toAPSInt(Bits);

  // Report undefined behaviour, stopping if required.
  const Expr *E = S.Current->getExpr(OpPC);
  QualType Type = E->getType();
  if (S.checkingForUndefinedBehavior()) {
    SmallString<32> Trunc;
    APResult.trunc(Result.bitWidth()).toString(Trunc, 10);
    auto Loc = E->getExprLoc();
    S.report(Loc, diag::warn_integer_constant_overflow)
        << Trunc << Type << E->getSourceRange();
    return true;
  }

  S.CCEDiag(E, diag::note_constexpr_overflow) << APResult << Type;
  return S.noteUndefinedBehavior();
}

/// 1) Pops a pointer from the stack
/// 2) Load the value from the pointer
/// 3) Writes the value increased by one back to the pointer
/// 4) Pushes the original (pre-inc) value on the stack.
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Inc(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckInitialized(S, OpPC, Ptr, AK_Increment))
    return false;

  return IncDecHelper<T, IncDecOp::Inc, PushVal::Yes>(S, OpPC, Ptr);
}

/// 1) Pops a pointer from the stack
/// 2) Load the value from the pointer
/// 3) Writes the value increased by one back to the pointer
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool IncPop(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckInitialized(S, OpPC, Ptr, AK_Increment))
    return false;

  return IncDecHelper<T, IncDecOp::Inc, PushVal::No>(S, OpPC, Ptr);
}

/// 1) Pops a pointer from the stack
/// 2) Load the value from the pointer
/// 3) Writes the value decreased by one back to the pointer
/// 4) Pushes the original (pre-dec) value on the stack.
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Dec(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckInitialized(S, OpPC, Ptr, AK_Decrement))
    return false;

  return IncDecHelper<T, IncDecOp::Dec, PushVal::Yes>(S, OpPC, Ptr);
}

/// 1) Pops a pointer from the stack
/// 2) Load the value from the pointer
/// 3) Writes the value decreased by one back to the pointer
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool DecPop(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckInitialized(S, OpPC, Ptr, AK_Decrement))
    return false;

  return IncDecHelper<T, IncDecOp::Dec, PushVal::No>(S, OpPC, Ptr);
}

template <IncDecOp Op, PushVal DoPush>
bool IncDecFloatHelper(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                       llvm::RoundingMode RM) {
  Floating Value = Ptr.deref<Floating>();
  Floating Result;

  if constexpr (DoPush == PushVal::Yes)
    S.Stk.push<Floating>(Value);

  llvm::APFloat::opStatus Status;
  if constexpr (Op == IncDecOp::Inc)
    Status = Floating::increment(Value, RM, &Result);
  else
    Status = Floating::decrement(Value, RM, &Result);

  Ptr.deref<Floating>() = Result;

  return CheckFloatResult(S, OpPC, Result, Status);
}

inline bool Incf(InterpState &S, CodePtr OpPC, llvm::RoundingMode RM) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckInitialized(S, OpPC, Ptr, AK_Increment))
    return false;

  return IncDecFloatHelper<IncDecOp::Inc, PushVal::Yes>(S, OpPC, Ptr, RM);
}

inline bool IncfPop(InterpState &S, CodePtr OpPC, llvm::RoundingMode RM) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckInitialized(S, OpPC, Ptr, AK_Increment))
    return false;

  return IncDecFloatHelper<IncDecOp::Inc, PushVal::No>(S, OpPC, Ptr, RM);
}

inline bool Decf(InterpState &S, CodePtr OpPC, llvm::RoundingMode RM) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckInitialized(S, OpPC, Ptr, AK_Decrement))
    return false;

  return IncDecFloatHelper<IncDecOp::Dec, PushVal::Yes>(S, OpPC, Ptr, RM);
}

inline bool DecfPop(InterpState &S, CodePtr OpPC, llvm::RoundingMode RM) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckInitialized(S, OpPC, Ptr, AK_Decrement))
    return false;

  return IncDecFloatHelper<IncDecOp::Dec, PushVal::No>(S, OpPC, Ptr, RM);
}

/// 1) Pops the value from the stack.
/// 2) Pushes the bitwise complemented value on the stack (~V).
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Comp(InterpState &S, CodePtr OpPC) {
  const T &Val = S.Stk.pop<T>();
  T Result;
  if (!T::comp(Val, &Result)) {
    S.Stk.push<T>(Result);
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// EQ, NE, GT, GE, LT, LE
//===----------------------------------------------------------------------===//

using CompareFn = llvm::function_ref<bool(ComparisonCategoryResult)>;

template <typename T>
bool CmpHelper(InterpState &S, CodePtr OpPC, CompareFn Fn) {
  using BoolT = PrimConv<PT_Bool>::T;
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();
  S.Stk.push<BoolT>(BoolT::from(Fn(LHS.compare(RHS))));
  return true;
}

template <typename T>
bool CmpHelperEQ(InterpState &S, CodePtr OpPC, CompareFn Fn) {
  return CmpHelper<T>(S, OpPC, Fn);
}

/// Function pointers cannot be compared in an ordered way.
template <>
inline bool CmpHelper<FunctionPointer>(InterpState &S, CodePtr OpPC,
                                       CompareFn Fn) {
  const auto &RHS = S.Stk.pop<FunctionPointer>();
  const auto &LHS = S.Stk.pop<FunctionPointer>();

  const SourceInfo &Loc = S.Current->getSource(OpPC);
  S.FFDiag(Loc, diag::note_constexpr_pointer_comparison_unspecified)
      << LHS.toDiagnosticString(S.getCtx())
      << RHS.toDiagnosticString(S.getCtx());
  return false;
}

template <>
inline bool CmpHelperEQ<FunctionPointer>(InterpState &S, CodePtr OpPC,
                                         CompareFn Fn) {
  const auto &RHS = S.Stk.pop<FunctionPointer>();
  const auto &LHS = S.Stk.pop<FunctionPointer>();
  S.Stk.push<Boolean>(Boolean::from(Fn(LHS.compare(RHS))));
  return true;
}

template <>
inline bool CmpHelper<Pointer>(InterpState &S, CodePtr OpPC, CompareFn Fn) {
  using BoolT = PrimConv<PT_Bool>::T;
  const Pointer &RHS = S.Stk.pop<Pointer>();
  const Pointer &LHS = S.Stk.pop<Pointer>();

  if (!Pointer::hasSameBase(LHS, RHS)) {
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_constexpr_pointer_comparison_unspecified)
        << LHS.toDiagnosticString(S.getCtx())
        << RHS.toDiagnosticString(S.getCtx());
    return false;
  } else {
    unsigned VL = LHS.getByteOffset();
    unsigned VR = RHS.getByteOffset();
    S.Stk.push<BoolT>(BoolT::from(Fn(Compare(VL, VR))));
    return true;
  }
}

template <>
inline bool CmpHelperEQ<Pointer>(InterpState &S, CodePtr OpPC, CompareFn Fn) {
  using BoolT = PrimConv<PT_Bool>::T;
  const Pointer &RHS = S.Stk.pop<Pointer>();
  const Pointer &LHS = S.Stk.pop<Pointer>();

  if (LHS.isZero() && RHS.isZero()) {
    S.Stk.push<BoolT>(BoolT::from(Fn(ComparisonCategoryResult::Equal)));
    return true;
  }

  if (!Pointer::hasSameBase(LHS, RHS)) {
    S.Stk.push<BoolT>(BoolT::from(Fn(ComparisonCategoryResult::Unordered)));
    return true;
  } else {
    unsigned VL = LHS.getByteOffset();
    unsigned VR = RHS.getByteOffset();

    // In our Pointer class, a pointer to an array and a pointer to the first
    // element in the same array are NOT equal. They have the same Base value,
    // but a different Offset. This is a pretty rare case, so we fix this here
    // by comparing pointers to the first elements.
    if (LHS.isArrayRoot())
      VL = LHS.atIndex(0).getByteOffset();
    if (RHS.isArrayRoot())
      VR = RHS.atIndex(0).getByteOffset();

    S.Stk.push<BoolT>(BoolT::from(Fn(Compare(VL, VR))));
    return true;
  }
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool EQ(InterpState &S, CodePtr OpPC) {
  return CmpHelperEQ<T>(S, OpPC, [](ComparisonCategoryResult R) {
    return R == ComparisonCategoryResult::Equal;
  });
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool CMP3(InterpState &S, CodePtr OpPC, const ComparisonCategoryInfo *CmpInfo) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();
  const Pointer &P = S.Stk.peek<Pointer>();

  ComparisonCategoryResult CmpResult = LHS.compare(RHS);
  if (CmpResult == ComparisonCategoryResult::Unordered) {
    // This should only happen with pointers.
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_constexpr_pointer_comparison_unspecified)
        << LHS.toDiagnosticString(S.getCtx())
        << RHS.toDiagnosticString(S.getCtx());
    return false;
  }

  assert(CmpInfo);
  const auto *CmpValueInfo = CmpInfo->getValueInfo(CmpResult);
  assert(CmpValueInfo);
  assert(CmpValueInfo->hasValidIntValue());
  APSInt IntValue = CmpValueInfo->getIntValue();
  return SetThreeWayComparisonField(S, OpPC, P, IntValue);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool NE(InterpState &S, CodePtr OpPC) {
  return CmpHelperEQ<T>(S, OpPC, [](ComparisonCategoryResult R) {
    return R != ComparisonCategoryResult::Equal;
  });
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool LT(InterpState &S, CodePtr OpPC) {
  return CmpHelper<T>(S, OpPC, [](ComparisonCategoryResult R) {
    return R == ComparisonCategoryResult::Less;
  });
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool LE(InterpState &S, CodePtr OpPC) {
  return CmpHelper<T>(S, OpPC, [](ComparisonCategoryResult R) {
    return R == ComparisonCategoryResult::Less ||
           R == ComparisonCategoryResult::Equal;
  });
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool GT(InterpState &S, CodePtr OpPC) {
  return CmpHelper<T>(S, OpPC, [](ComparisonCategoryResult R) {
    return R == ComparisonCategoryResult::Greater;
  });
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool GE(InterpState &S, CodePtr OpPC) {
  return CmpHelper<T>(S, OpPC, [](ComparisonCategoryResult R) {
    return R == ComparisonCategoryResult::Greater ||
           R == ComparisonCategoryResult::Equal;
  });
}

//===----------------------------------------------------------------------===//
// InRange
//===----------------------------------------------------------------------===//

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InRange(InterpState &S, CodePtr OpPC) {
  const T RHS = S.Stk.pop<T>();
  const T LHS = S.Stk.pop<T>();
  const T Value = S.Stk.pop<T>();

  S.Stk.push<bool>(LHS <= Value && Value <= RHS);
  return true;
}

//===----------------------------------------------------------------------===//
// Dup, Pop, Test
//===----------------------------------------------------------------------===//

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Dup(InterpState &S, CodePtr OpPC) {
  S.Stk.push<T>(S.Stk.peek<T>());
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Pop(InterpState &S, CodePtr OpPC) {
  S.Stk.pop<T>();
  return true;
}

//===----------------------------------------------------------------------===//
// Const
//===----------------------------------------------------------------------===//

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Const(InterpState &S, CodePtr OpPC, const T &Arg) {
  S.Stk.push<T>(Arg);
  return true;
}

//===----------------------------------------------------------------------===//
// Get/Set Local/Param/Global/This
//===----------------------------------------------------------------------===//

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool GetLocal(InterpState &S, CodePtr OpPC, uint32_t I) {
  const Pointer &Ptr = S.Current->getLocalPointer(I);
  if (!CheckLoad(S, OpPC, Ptr))
    return false;
  S.Stk.push<T>(Ptr.deref<T>());
  return true;
}

/// 1) Pops the value from the stack.
/// 2) Writes the value to the local variable with the
///    given offset.
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool SetLocal(InterpState &S, CodePtr OpPC, uint32_t I) {
  S.Current->setLocal<T>(I, S.Stk.pop<T>());
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool GetParam(InterpState &S, CodePtr OpPC, uint32_t I) {
  if (S.checkingPotentialConstantExpression()) {
    return false;
  }
  S.Stk.push<T>(S.Current->getParam<T>(I));
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool SetParam(InterpState &S, CodePtr OpPC, uint32_t I) {
  S.Current->setParam<T>(I, S.Stk.pop<T>());
  return true;
}

/// 1) Peeks a pointer on the stack
/// 2) Pushes the value of the pointer's field on the stack
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool GetField(InterpState &S, CodePtr OpPC, uint32_t I) {
  const Pointer &Obj = S.Stk.peek<Pointer>();
  if (!CheckNull(S, OpPC, Obj, CSK_Field))
      return false;
  if (!CheckRange(S, OpPC, Obj, CSK_Field))
    return false;
  const Pointer &Field = Obj.atField(I);
  if (!CheckLoad(S, OpPC, Field))
    return false;
  S.Stk.push<T>(Field.deref<T>());
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool SetField(InterpState &S, CodePtr OpPC, uint32_t I) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Obj = S.Stk.peek<Pointer>();
  if (!CheckNull(S, OpPC, Obj, CSK_Field))
    return false;
  if (!CheckRange(S, OpPC, Obj, CSK_Field))
    return false;
  const Pointer &Field = Obj.atField(I);
  if (!CheckStore(S, OpPC, Field))
    return false;
  Field.initialize();
  Field.deref<T>() = Value;
  return true;
}

/// 1) Pops a pointer from the stack
/// 2) Pushes the value of the pointer's field on the stack
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool GetFieldPop(InterpState &S, CodePtr OpPC, uint32_t I) {
  const Pointer &Obj = S.Stk.pop<Pointer>();
  if (!CheckNull(S, OpPC, Obj, CSK_Field))
    return false;
  if (!CheckRange(S, OpPC, Obj, CSK_Field))
    return false;
  const Pointer &Field = Obj.atField(I);
  if (!CheckLoad(S, OpPC, Field))
    return false;
  S.Stk.push<T>(Field.deref<T>());
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool GetThisField(InterpState &S, CodePtr OpPC, uint32_t I) {
  if (S.checkingPotentialConstantExpression())
    return false;
  const Pointer &This = S.Current->getThis();
  if (!CheckThis(S, OpPC, This))
    return false;
  const Pointer &Field = This.atField(I);
  if (!CheckLoad(S, OpPC, Field))
    return false;
  S.Stk.push<T>(Field.deref<T>());
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool SetThisField(InterpState &S, CodePtr OpPC, uint32_t I) {
  if (S.checkingPotentialConstantExpression())
    return false;
  const T &Value = S.Stk.pop<T>();
  const Pointer &This = S.Current->getThis();
  if (!CheckThis(S, OpPC, This))
    return false;
  const Pointer &Field = This.atField(I);
  if (!CheckStore(S, OpPC, Field))
    return false;
  Field.deref<T>() = Value;
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool GetGlobal(InterpState &S, CodePtr OpPC, uint32_t I) {
  const Pointer &Ptr = S.P.getPtrGlobal(I);
  if (!CheckConstant(S, OpPC, Ptr.getFieldDesc()))
    return false;
  if (Ptr.isExtern())
    return false;

  // If a global variable is uninitialized, that means the initializer we've
  // compiled for it wasn't a constant expression. Diagnose that.
  if (!CheckGlobalInitialized(S, OpPC, Ptr))
    return false;

  S.Stk.push<T>(Ptr.deref<T>());
  return true;
}

/// Same as GetGlobal, but without the checks.
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool GetGlobalUnchecked(InterpState &S, CodePtr OpPC, uint32_t I) {
  auto *B = S.P.getGlobal(I);
  S.Stk.push<T>(B->deref<T>());
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool SetGlobal(InterpState &S, CodePtr OpPC, uint32_t I) {
  // TODO: emit warning.
  return false;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitGlobal(InterpState &S, CodePtr OpPC, uint32_t I) {
  const Pointer &P = S.P.getGlobal(I);
  P.deref<T>() = S.Stk.pop<T>();
  P.initialize();
  return true;
}

/// 1) Converts the value on top of the stack to an APValue
/// 2) Sets that APValue on \Temp
/// 3) Initialized global with index \I with that
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitGlobalTemp(InterpState &S, CodePtr OpPC, uint32_t I,
                    const LifetimeExtendedTemporaryDecl *Temp) {
  assert(Temp);
  const T Value = S.Stk.peek<T>();
  APValue APV = Value.toAPValue();
  APValue *Cached = Temp->getOrCreateValue(true);
  *Cached = APV;

  const Pointer &P = S.P.getGlobal(I);
  P.deref<T>() = S.Stk.pop<T>();
  P.initialize();

  return true;
}

/// 1) Converts the value on top of the stack to an APValue
/// 2) Sets that APValue on \Temp
/// 3) Initialized global with index \I with that
inline bool InitGlobalTempComp(InterpState &S, CodePtr OpPC,
                               const LifetimeExtendedTemporaryDecl *Temp) {
  assert(Temp);
  const Pointer &P = S.Stk.peek<Pointer>();
  APValue *Cached = Temp->getOrCreateValue(true);

  if (std::optional<APValue> APV = P.toRValue(S.getCtx())) {
    *Cached = *APV;
    return true;
  }

  return false;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitThisField(InterpState &S, CodePtr OpPC, uint32_t I) {
  if (S.checkingPotentialConstantExpression())
    return false;
  const Pointer &This = S.Current->getThis();
  if (!CheckThis(S, OpPC, This))
    return false;
  const Pointer &Field = This.atField(I);
  Field.deref<T>() = S.Stk.pop<T>();
  Field.initialize();
  return true;
}

// FIXME: The Field pointer here is too much IMO and we could instead just
// pass an Offset + BitWidth pair.
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitThisBitField(InterpState &S, CodePtr OpPC, const Record::Field *F,
                      uint32_t FieldOffset) {
  assert(F->isBitField());
  if (S.checkingPotentialConstantExpression())
    return false;
  const Pointer &This = S.Current->getThis();
  if (!CheckThis(S, OpPC, This))
    return false;
  const Pointer &Field = This.atField(FieldOffset);
  const auto &Value = S.Stk.pop<T>();
  Field.deref<T>() = Value.truncate(F->Decl->getBitWidthValue(S.getCtx()));
  Field.initialize();
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitThisFieldActive(InterpState &S, CodePtr OpPC, uint32_t I) {
  if (S.checkingPotentialConstantExpression())
    return false;
  const Pointer &This = S.Current->getThis();
  if (!CheckThis(S, OpPC, This))
    return false;
  const Pointer &Field = This.atField(I);
  Field.deref<T>() = S.Stk.pop<T>();
  Field.activate();
  Field.initialize();
  return true;
}

/// 1) Pops the value from the stack
/// 2) Peeks a pointer from the stack
/// 3) Pushes the value to field I of the pointer on the stack
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitField(InterpState &S, CodePtr OpPC, uint32_t I) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Field = S.Stk.peek<Pointer>().atField(I);
  Field.deref<T>() = Value;
  Field.activate();
  Field.initialize();
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitBitField(InterpState &S, CodePtr OpPC, const Record::Field *F) {
  assert(F->isBitField());
  const T &Value = S.Stk.pop<T>();
  const Pointer &Field = S.Stk.peek<Pointer>().atField(F->Offset);
  Field.deref<T>() = Value.truncate(F->Decl->getBitWidthValue(S.getCtx()));
  Field.activate();
  Field.initialize();
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitFieldActive(InterpState &S, CodePtr OpPC, uint32_t I) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  const Pointer &Field = Ptr.atField(I);
  Field.deref<T>() = Value;
  Field.activate();
  Field.initialize();
  return true;
}

//===----------------------------------------------------------------------===//
// GetPtr Local/Param/Global/Field/This
//===----------------------------------------------------------------------===//

inline bool GetPtrLocal(InterpState &S, CodePtr OpPC, uint32_t I) {
  S.Stk.push<Pointer>(S.Current->getLocalPointer(I));
  return true;
}

inline bool GetPtrParam(InterpState &S, CodePtr OpPC, uint32_t I) {
  if (S.checkingPotentialConstantExpression()) {
    return false;
  }
  S.Stk.push<Pointer>(S.Current->getParamPointer(I));
  return true;
}

inline bool GetPtrGlobal(InterpState &S, CodePtr OpPC, uint32_t I) {
  S.Stk.push<Pointer>(S.P.getPtrGlobal(I));
  return true;
}

/// 1) Pops a Pointer from the stack
/// 2) Pushes Pointer.atField(Off) on the stack
inline bool GetPtrField(InterpState &S, CodePtr OpPC, uint32_t Off) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (S.inConstantContext() && !CheckNull(S, OpPC, Ptr, CSK_Field))
    return false;
  if (!CheckExtern(S, OpPC, Ptr))
    return false;
  if (!CheckRange(S, OpPC, Ptr, CSK_Field))
    return false;
  if (!CheckSubobject(S, OpPC, Ptr, CSK_Field))
    return false;

  S.Stk.push<Pointer>(Ptr.atField(Off));
  return true;
}

inline bool GetPtrThisField(InterpState &S, CodePtr OpPC, uint32_t Off) {
  if (S.checkingPotentialConstantExpression())
    return false;
  const Pointer &This = S.Current->getThis();
  if (!CheckThis(S, OpPC, This))
    return false;
  S.Stk.push<Pointer>(This.atField(Off));
  return true;
}

inline bool GetPtrActiveField(InterpState &S, CodePtr OpPC, uint32_t Off) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckNull(S, OpPC, Ptr, CSK_Field))
    return false;
  if (!CheckRange(S, OpPC, Ptr, CSK_Field))
    return false;
  Pointer Field = Ptr.atField(Off);
  Ptr.deactivate();
  Field.activate();
  S.Stk.push<Pointer>(std::move(Field));
  return true;
}

inline bool GetPtrActiveThisField(InterpState &S, CodePtr OpPC, uint32_t Off) {
 if (S.checkingPotentialConstantExpression())
    return false;
  const Pointer &This = S.Current->getThis();
  if (!CheckThis(S, OpPC, This))
    return false;
  Pointer Field = This.atField(Off);
  This.deactivate();
  Field.activate();
  S.Stk.push<Pointer>(std::move(Field));
  return true;
}

inline bool GetPtrDerivedPop(InterpState &S, CodePtr OpPC, uint32_t Off) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckNull(S, OpPC, Ptr, CSK_Derived))
    return false;
  if (!CheckSubobject(S, OpPC, Ptr, CSK_Derived))
    return false;
  S.Stk.push<Pointer>(Ptr.atFieldSub(Off));
  return true;
}

inline bool GetPtrBase(InterpState &S, CodePtr OpPC, uint32_t Off) {
  const Pointer &Ptr = S.Stk.peek<Pointer>();
  if (!CheckNull(S, OpPC, Ptr, CSK_Base))
    return false;
  if (!CheckSubobject(S, OpPC, Ptr, CSK_Base))
    return false;
  S.Stk.push<Pointer>(Ptr.atField(Off));
  return true;
}

inline bool GetPtrBasePop(InterpState &S, CodePtr OpPC, uint32_t Off) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckNull(S, OpPC, Ptr, CSK_Base))
    return false;
  if (!CheckSubobject(S, OpPC, Ptr, CSK_Base))
    return false;
  S.Stk.push<Pointer>(Ptr.atField(Off));
  return true;
}

inline bool GetPtrThisBase(InterpState &S, CodePtr OpPC, uint32_t Off) {
  if (S.checkingPotentialConstantExpression())
    return false;
  const Pointer &This = S.Current->getThis();
  if (!CheckThis(S, OpPC, This))
    return false;
  S.Stk.push<Pointer>(This.atField(Off));
  return true;
}

inline bool InitPtrPop(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  Ptr.initialize();
  return true;
}

inline bool InitPtr(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.peek<Pointer>();
  Ptr.initialize();
  return true;
}

inline bool VirtBaseHelper(InterpState &S, CodePtr OpPC, const RecordDecl *Decl,
                           const Pointer &Ptr) {
  Pointer Base = Ptr;
  while (Base.isBaseClass())
    Base = Base.getBase();

  auto *Field = Base.getRecord()->getVirtualBase(Decl);
  S.Stk.push<Pointer>(Base.atField(Field->Offset));
  return true;
}

inline bool GetPtrVirtBase(InterpState &S, CodePtr OpPC, const RecordDecl *D) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckNull(S, OpPC, Ptr, CSK_Base))
    return false;
  return VirtBaseHelper(S, OpPC, D, Ptr);
}

inline bool GetPtrThisVirtBase(InterpState &S, CodePtr OpPC,
                               const RecordDecl *D) {
  if (S.checkingPotentialConstantExpression())
    return false;
  const Pointer &This = S.Current->getThis();
  if (!CheckThis(S, OpPC, This))
    return false;
  return VirtBaseHelper(S, OpPC, D, S.Current->getThis());
}

//===----------------------------------------------------------------------===//
// Load, Store, Init
//===----------------------------------------------------------------------===//

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Load(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.peek<Pointer>();
  if (!CheckLoad(S, OpPC, Ptr))
    return false;
  S.Stk.push<T>(Ptr.deref<T>());
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool LoadPop(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckLoad(S, OpPC, Ptr))
    return false;
  S.Stk.push<T>(Ptr.deref<T>());
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Store(InterpState &S, CodePtr OpPC) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.peek<Pointer>();
  if (!CheckStore(S, OpPC, Ptr))
    return false;
  if (Ptr.canBeInitialized())
    Ptr.initialize();
  Ptr.deref<T>() = Value;
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool StorePop(InterpState &S, CodePtr OpPC) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckStore(S, OpPC, Ptr))
    return false;
  if (Ptr.canBeInitialized())
    Ptr.initialize();
  Ptr.deref<T>() = Value;
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool StoreBitField(InterpState &S, CodePtr OpPC) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.peek<Pointer>();
  if (!CheckStore(S, OpPC, Ptr))
    return false;
  if (Ptr.canBeInitialized())
    Ptr.initialize();
  if (const auto *FD = Ptr.getField())
    Ptr.deref<T>() = Value.truncate(FD->getBitWidthValue(S.getCtx()));
  else
    Ptr.deref<T>() = Value;
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool StoreBitFieldPop(InterpState &S, CodePtr OpPC) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckStore(S, OpPC, Ptr))
    return false;
  if (Ptr.canBeInitialized())
    Ptr.initialize();
  if (const auto *FD = Ptr.getField())
    Ptr.deref<T>() = Value.truncate(FD->getBitWidthValue(S.getCtx()));
  else
    Ptr.deref<T>() = Value;
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitPop(InterpState &S, CodePtr OpPC) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckInit(S, OpPC, Ptr))
    return false;
  Ptr.initialize();
  new (&Ptr.deref<T>()) T(Value);
  return true;
}

/// 1) Pops the value from the stack
/// 2) Peeks a pointer and gets its index \Idx
/// 3) Sets the value on the pointer, leaving the pointer on the stack.
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitElem(InterpState &S, CodePtr OpPC, uint32_t Idx) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.peek<Pointer>().atIndex(Idx);
  if (!CheckInit(S, OpPC, Ptr))
    return false;
  Ptr.initialize();
  new (&Ptr.deref<T>()) T(Value);
  return true;
}

/// The same as InitElem, but pops the pointer as well.
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitElemPop(InterpState &S, CodePtr OpPC, uint32_t Idx) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.pop<Pointer>().atIndex(Idx);
  if (!CheckInit(S, OpPC, Ptr))
    return false;
  Ptr.initialize();
  new (&Ptr.deref<T>()) T(Value);
  return true;
}

//===----------------------------------------------------------------------===//
// AddOffset, SubOffset
//===----------------------------------------------------------------------===//

template <class T, ArithOp Op>
bool OffsetHelper(InterpState &S, CodePtr OpPC, const T &Offset,
                  const Pointer &Ptr) {
  if (!CheckRange(S, OpPC, Ptr, CSK_ArrayToPointer))
    return false;

  // A zero offset does not change the pointer.
  if (Offset.isZero()) {
    S.Stk.push<Pointer>(Ptr);
    return true;
  }

  if (!CheckNull(S, OpPC, Ptr, CSK_ArrayIndex))
    return false;

  // Arrays of unknown bounds cannot have pointers into them.
  if (!CheckArray(S, OpPC, Ptr))
    return false;

  // Get a version of the index comparable to the type.
  T Index = T::from(Ptr.getIndex(), Offset.bitWidth());
  // Compute the largest index into the array.
  T MaxIndex = T::from(Ptr.getNumElems(), Offset.bitWidth());

  bool Invalid = false;
  // Helper to report an invalid offset, computed as APSInt.
  auto DiagInvalidOffset = [&]() -> void {
    const unsigned Bits = Offset.bitWidth();
    APSInt APOffset(Offset.toAPSInt().extend(Bits + 2), false);
    APSInt APIndex(Index.toAPSInt().extend(Bits + 2), false);
    APSInt NewIndex =
        (Op == ArithOp::Add) ? (APIndex + APOffset) : (APIndex - APOffset);
    S.CCEDiag(S.Current->getSource(OpPC), diag::note_constexpr_array_index)
        << NewIndex
        << /*array*/ static_cast<int>(!Ptr.inArray())
        << static_cast<unsigned>(MaxIndex);
    Invalid = true;
  };

  T MaxOffset = T::from(MaxIndex - Index, Offset.bitWidth());
  if constexpr (Op == ArithOp::Add) {
    // If the new offset would be negative, bail out.
    if (Offset.isNegative() && (Offset.isMin() || -Offset > Index))
      DiagInvalidOffset();

    // If the new offset would be out of bounds, bail out.
    if (Offset.isPositive() && Offset > MaxOffset)
      DiagInvalidOffset();
  } else {
    // If the new offset would be negative, bail out.
    if (Offset.isPositive() && Index < Offset)
      DiagInvalidOffset();

    // If the new offset would be out of bounds, bail out.
    if (Offset.isNegative() && (Offset.isMin() || -Offset > MaxOffset))
      DiagInvalidOffset();
  }

  if (Invalid && !Ptr.isDummy() && S.getLangOpts().CPlusPlus)
    return false;

  // Offset is valid - compute it on unsigned.
  int64_t WideIndex = static_cast<int64_t>(Index);
  int64_t WideOffset = static_cast<int64_t>(Offset);
  int64_t Result;
  if constexpr (Op == ArithOp::Add)
    Result = WideIndex + WideOffset;
  else
    Result = WideIndex - WideOffset;

  S.Stk.push<Pointer>(Ptr.atIndex(static_cast<unsigned>(Result)));
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool AddOffset(InterpState &S, CodePtr OpPC) {
  const T &Offset = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  return OffsetHelper<T, ArithOp::Add>(S, OpPC, Offset, Ptr);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool SubOffset(InterpState &S, CodePtr OpPC) {
  const T &Offset = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  return OffsetHelper<T, ArithOp::Sub>(S, OpPC, Offset, Ptr);
}

template <ArithOp Op>
static inline bool IncDecPtrHelper(InterpState &S, CodePtr OpPC,
                                   const Pointer &Ptr) {
  if (Ptr.isDummy())
    return false;

  using OneT = Integral<8, false>;

  const Pointer &P = Ptr.deref<Pointer>();
  if (!CheckNull(S, OpPC, P, CSK_ArrayIndex))
    return false;

  // Get the current value on the stack.
  S.Stk.push<Pointer>(P);

  // Now the current Ptr again and a constant 1.
  OneT One = OneT::from(1);
  if (!OffsetHelper<OneT, Op>(S, OpPC, One, P))
    return false;

  // Store the new value.
  Ptr.deref<Pointer>() = S.Stk.pop<Pointer>();
  return true;
}

static inline bool IncPtr(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckInitialized(S, OpPC, Ptr, AK_Increment))
    return false;

  return IncDecPtrHelper<ArithOp::Add>(S, OpPC, Ptr);
}

static inline bool DecPtr(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckInitialized(S, OpPC, Ptr, AK_Decrement))
    return false;

  return IncDecPtrHelper<ArithOp::Sub>(S, OpPC, Ptr);
}

/// 1) Pops a Pointer from the stack.
/// 2) Pops another Pointer from the stack.
/// 3) Pushes the different of the indices of the two pointers on the stack.
template <PrimType Name, class T = typename PrimConv<Name>::T>
inline bool SubPtr(InterpState &S, CodePtr OpPC) {
  const Pointer &LHS = S.Stk.pop<Pointer>();
  const Pointer &RHS = S.Stk.pop<Pointer>();

  if (!Pointer::hasSameBase(LHS, RHS) && S.getLangOpts().CPlusPlus) {
    // TODO: Diagnose.
    return false;
  }

  T A = T::from(LHS.getIndex());
  T B = T::from(RHS.getIndex());
  return AddSubMulHelper<T, T::sub, std::minus>(S, OpPC, A.bitWidth(), A, B);
}

//===----------------------------------------------------------------------===//
// Destroy
//===----------------------------------------------------------------------===//

inline bool Destroy(InterpState &S, CodePtr OpPC, uint32_t I) {
  S.Current->destroy(I);
  return true;
}

//===----------------------------------------------------------------------===//
// Cast, CastFP
//===----------------------------------------------------------------------===//

template <PrimType TIn, PrimType TOut> bool Cast(InterpState &S, CodePtr OpPC) {
  using T = typename PrimConv<TIn>::T;
  using U = typename PrimConv<TOut>::T;
  S.Stk.push<U>(U::from(S.Stk.pop<T>()));
  return true;
}

/// 1) Pops a Floating from the stack.
/// 2) Pushes a new floating on the stack that uses the given semantics.
inline bool CastFP(InterpState &S, CodePtr OpPC, const llvm::fltSemantics *Sem,
            llvm::RoundingMode RM) {
  Floating F = S.Stk.pop<Floating>();
  Floating Result = F.toSemantics(Sem, RM);
  S.Stk.push<Floating>(Result);
  return true;
}

/// Like Cast(), but we cast to an arbitrary-bitwidth integral, so we need
/// to know what bitwidth the result should be.
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool CastAP(InterpState &S, CodePtr OpPC, uint32_t BitWidth) {
  S.Stk.push<IntegralAP<false>>(
      IntegralAP<false>::from(S.Stk.pop<T>(), BitWidth));
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool CastAPS(InterpState &S, CodePtr OpPC, uint32_t BitWidth) {
  S.Stk.push<IntegralAP<true>>(
      IntegralAP<true>::from(S.Stk.pop<T>(), BitWidth));
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool CastIntegralFloating(InterpState &S, CodePtr OpPC,
                          const llvm::fltSemantics *Sem,
                          llvm::RoundingMode RM) {
  const T &From = S.Stk.pop<T>();
  APSInt FromAP = From.toAPSInt();
  Floating Result;

  auto Status = Floating::fromIntegral(FromAP, *Sem, RM, Result);
  S.Stk.push<Floating>(Result);

  return CheckFloatResult(S, OpPC, Result, Status);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool CastFloatingIntegral(InterpState &S, CodePtr OpPC) {
  const Floating &F = S.Stk.pop<Floating>();

  if constexpr (std::is_same_v<T, Boolean>) {
    S.Stk.push<T>(T(F.isNonZero()));
    return true;
  } else {
    APSInt Result(std::max(8u, T::bitWidth()),
                  /*IsUnsigned=*/!T::isSigned());
    auto Status = F.convertToInteger(Result);

    // Float-to-Integral overflow check.
    if ((Status & APFloat::opStatus::opInvalidOp) && F.isFinite()) {
      const Expr *E = S.Current->getExpr(OpPC);
      QualType Type = E->getType();

      S.CCEDiag(E, diag::note_constexpr_overflow) << F.getAPFloat() << Type;
      if (S.noteUndefinedBehavior()) {
        S.Stk.push<T>(T(Result));
        return true;
      }
      return false;
    }

    S.Stk.push<T>(T(Result));
    return CheckFloatResult(S, OpPC, F, Status);
  }
}

static inline bool CastFloatingIntegralAP(InterpState &S, CodePtr OpPC,
                                          uint32_t BitWidth) {
  const Floating &F = S.Stk.pop<Floating>();

  APSInt Result(BitWidth, /*IsUnsigned=*/true);
  auto Status = F.convertToInteger(Result);

  // Float-to-Integral overflow check.
  if ((Status & APFloat::opStatus::opInvalidOp) && F.isFinite()) {
    const Expr *E = S.Current->getExpr(OpPC);
    QualType Type = E->getType();

    S.CCEDiag(E, diag::note_constexpr_overflow) << F.getAPFloat() << Type;
    return S.noteUndefinedBehavior();
  }

  S.Stk.push<IntegralAP<true>>(IntegralAP<true>(Result));
  return CheckFloatResult(S, OpPC, F, Status);
}

static inline bool CastFloatingIntegralAPS(InterpState &S, CodePtr OpPC,
                                           uint32_t BitWidth) {
  const Floating &F = S.Stk.pop<Floating>();

  APSInt Result(BitWidth, /*IsUnsigned=*/false);
  auto Status = F.convertToInteger(Result);

  // Float-to-Integral overflow check.
  if ((Status & APFloat::opStatus::opInvalidOp) && F.isFinite()) {
    const Expr *E = S.Current->getExpr(OpPC);
    QualType Type = E->getType();

    S.CCEDiag(E, diag::note_constexpr_overflow) << F.getAPFloat() << Type;
    return S.noteUndefinedBehavior();
  }

  S.Stk.push<IntegralAP<true>>(IntegralAP<true>(Result));
  return CheckFloatResult(S, OpPC, F, Status);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool CastPointerIntegral(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckPotentialReinterpretCast(S, OpPC, Ptr))
    return false;

  S.Stk.push<T>(T::from(Ptr.getIntegerRepresentation()));
  return true;
}

//===----------------------------------------------------------------------===//
// Zero, Nullptr
//===----------------------------------------------------------------------===//

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Zero(InterpState &S, CodePtr OpPC) {
  S.Stk.push<T>(T::zero());
  return true;
}

static inline bool ZeroIntAP(InterpState &S, CodePtr OpPC, uint32_t BitWidth) {
  S.Stk.push<IntegralAP<false>>(IntegralAP<false>::zero(BitWidth));
  return true;
}

static inline bool ZeroIntAPS(InterpState &S, CodePtr OpPC, uint32_t BitWidth) {
  S.Stk.push<IntegralAP<true>>(IntegralAP<true>::zero(BitWidth));
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
inline bool Null(InterpState &S, CodePtr OpPC) {
  S.Stk.push<T>();
  return true;
}

//===----------------------------------------------------------------------===//
// This, ImplicitThis
//===----------------------------------------------------------------------===//

inline bool This(InterpState &S, CodePtr OpPC) {
  // Cannot read 'this' in this mode.
  if (S.checkingPotentialConstantExpression()) {
    return false;
  }

  const Pointer &This = S.Current->getThis();
  if (!CheckThis(S, OpPC, This))
    return false;

  S.Stk.push<Pointer>(This);
  return true;
}

inline bool RVOPtr(InterpState &S, CodePtr OpPC) {
  assert(S.Current->getFunction()->hasRVO());
  if (S.checkingPotentialConstantExpression())
    return false;
  S.Stk.push<Pointer>(S.Current->getRVOPtr());
  return true;
}

//===----------------------------------------------------------------------===//
// Shr, Shl
//===----------------------------------------------------------------------===//

template <PrimType NameL, PrimType NameR>
inline bool Shr(InterpState &S, CodePtr OpPC) {
  using LT = typename PrimConv<NameL>::T;
  using RT = typename PrimConv<NameR>::T;
  const auto &RHS = S.Stk.pop<RT>();
  const auto &LHS = S.Stk.pop<LT>();
  const unsigned Bits = LHS.bitWidth();

  if (!CheckShift(S, OpPC, LHS, RHS, Bits))
    return false;

  typename LT::AsUnsigned R;
  LT::AsUnsigned::shiftRight(LT::AsUnsigned::from(LHS),
                             LT::AsUnsigned::from(RHS), Bits, &R);
  S.Stk.push<LT>(LT::from(R));

  return true;
}

template <PrimType NameL, PrimType NameR>
inline bool Shl(InterpState &S, CodePtr OpPC) {
  using LT = typename PrimConv<NameL>::T;
  using RT = typename PrimConv<NameR>::T;
  const auto &RHS = S.Stk.pop<RT>();
  const auto &LHS = S.Stk.pop<LT>();
  const unsigned Bits = LHS.bitWidth();

  if (!CheckShift(S, OpPC, LHS, RHS, Bits))
    return false;

  typename LT::AsUnsigned R;
  LT::AsUnsigned::shiftLeft(LT::AsUnsigned::from(LHS),
                            LT::AsUnsigned::from(RHS, Bits), Bits, &R);
  S.Stk.push<LT>(LT::from(R));
  return true;
}

//===----------------------------------------------------------------------===//
// NoRet
//===----------------------------------------------------------------------===//

inline bool NoRet(InterpState &S, CodePtr OpPC) {
  SourceLocation EndLoc = S.Current->getCallee()->getEndLoc();
  S.FFDiag(EndLoc, diag::note_constexpr_no_return);
  return false;
}

//===----------------------------------------------------------------------===//
// NarrowPtr, ExpandPtr
//===----------------------------------------------------------------------===//

inline bool NarrowPtr(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  S.Stk.push<Pointer>(Ptr.narrow());
  return true;
}

inline bool ExpandPtr(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  S.Stk.push<Pointer>(Ptr.expand());
  return true;
}

// 1) Pops an integral value from the stack
// 2) Peeks a pointer
// 3) Pushes a new pointer that's a narrowed array
//   element of the peeked pointer with the value
//   from 1) added as offset.
//
// This leaves the original pointer on the stack and pushes a new one
// with the offset applied and narrowed.
template <PrimType Name, class T = typename PrimConv<Name>::T>
inline bool ArrayElemPtr(InterpState &S, CodePtr OpPC) {
  const T &Offset = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.peek<Pointer>();

  if (!CheckDummy(S, OpPC, Ptr))
    return false;

  if (!OffsetHelper<T, ArithOp::Add>(S, OpPC, Offset, Ptr))
    return false;

  return NarrowPtr(S, OpPC);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
inline bool ArrayElemPtrPop(InterpState &S, CodePtr OpPC) {
  const T &Offset = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!CheckDummy(S, OpPC, Ptr))
    return false;

  if (!OffsetHelper<T, ArithOp::Add>(S, OpPC, Offset, Ptr))
    return false;

  return NarrowPtr(S, OpPC);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
inline bool ArrayElemPop(InterpState &S, CodePtr OpPC, uint32_t Index) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  S.Stk.push<T>(Ptr.atIndex(Index).deref<T>());
  return true;
}

/// Just takes a pointer and checks if it's an incomplete
/// array type.
inline bool ArrayDecay(InterpState &S, CodePtr OpPC) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  if (!Ptr.isUnknownSizeArray()) {
    S.Stk.push<Pointer>(Ptr.atIndex(0));
    return true;
  }

  const SourceInfo &E = S.Current->getSource(OpPC);
  S.FFDiag(E, diag::note_constexpr_unsupported_unsized_array);

  return false;
}

inline bool Call(InterpState &S, CodePtr OpPC, const Function *Func) {
  if (Func->hasThisPointer()) {
    size_t ThisOffset =
        Func->getArgSize() - (Func->hasRVO() ? primSize(PT_Ptr) : 0);

    const Pointer &ThisPtr = S.Stk.peek<Pointer>(ThisOffset);

    // If the current function is a lambda static invoker and
    // the function we're about to call is a lambda call operator,
    // skip the CheckInvoke, since the ThisPtr is a null pointer
    // anyway.
    if (!(S.Current->getFunction() &&
          S.Current->getFunction()->isLambdaStaticInvoker() &&
          Func->isLambdaCallOperator())) {
      if (!CheckInvoke(S, OpPC, ThisPtr))
        return false;
    }

    if (S.checkingPotentialConstantExpression())
      return false;
  }

  if (!CheckCallable(S, OpPC, Func))
    return false;

  if (!CheckCallDepth(S, OpPC))
    return false;

  auto NewFrame = std::make_unique<InterpFrame>(S, Func, OpPC);
  InterpFrame *FrameBefore = S.Current;
  S.Current = NewFrame.get();

  APValue CallResult;
  // Note that we cannot assert(CallResult.hasValue()) here since
  // Ret() above only sets the APValue if the curent frame doesn't
  // have a caller set.
  if (Interpret(S, CallResult)) {
    NewFrame.release(); // Frame was delete'd already.
    assert(S.Current == FrameBefore);
    return true;
  }

  // Interpreting the function failed somehow. Reset to
  // previous state.
  S.Current = FrameBefore;
  return false;
}

inline bool CallVirt(InterpState &S, CodePtr OpPC, const Function *Func) {
  assert(Func->hasThisPointer());
  assert(Func->isVirtual());
  size_t ThisOffset =
      Func->getArgSize() - (Func->hasRVO() ? primSize(PT_Ptr) : 0);
  Pointer &ThisPtr = S.Stk.peek<Pointer>(ThisOffset);

  const CXXRecordDecl *DynamicDecl =
      ThisPtr.getDeclDesc()->getType()->getAsCXXRecordDecl();
  const auto *StaticDecl = cast<CXXRecordDecl>(Func->getParentDecl());
  const auto *InitialFunction = cast<CXXMethodDecl>(Func->getDecl());
  const CXXMethodDecl *Overrider = S.getContext().getOverridingFunction(
      DynamicDecl, StaticDecl, InitialFunction);

  if (Overrider != InitialFunction) {
    // DR1872: An instantiated virtual constexpr function can't be called in a
    // constant expression (prior to C++20). We can still constant-fold such a
    // call.
    if (!S.getLangOpts().CPlusPlus20 && Overrider->isVirtual()) {
      const Expr *E = S.Current->getExpr(OpPC);
      S.CCEDiag(E, diag::note_constexpr_virtual_call) << E->getSourceRange();
    }

    Func = S.getContext().getOrCreateFunction(Overrider);

    const CXXRecordDecl *ThisFieldDecl =
        ThisPtr.getFieldDesc()->getType()->getAsCXXRecordDecl();
    if (Func->getParentDecl()->isDerivedFrom(ThisFieldDecl)) {
      // If the function we call is further DOWN the hierarchy than the
      // FieldDesc of our pointer, just get the DeclDesc instead, which
      // is the furthest we might go up in the hierarchy.
      ThisPtr = ThisPtr.getDeclPtr();
    }
  }

  return Call(S, OpPC, Func);
}

inline bool CallBI(InterpState &S, CodePtr &PC, const Function *Func,
                   const CallExpr *CE) {
  auto NewFrame = std::make_unique<InterpFrame>(S, Func, PC);

  InterpFrame *FrameBefore = S.Current;
  S.Current = NewFrame.get();

  if (InterpretBuiltin(S, PC, Func, CE)) {
    NewFrame.release();
    return true;
  }
  S.Current = FrameBefore;
  return false;
}

inline bool CallPtr(InterpState &S, CodePtr OpPC) {
  const FunctionPointer &FuncPtr = S.Stk.pop<FunctionPointer>();

  const Function *F = FuncPtr.getFunction();
  if (!F || !F->isConstexpr())
    return false;

  if (F->isVirtual())
    return CallVirt(S, OpPC, F);

  return Call(S, OpPC, F);
}

inline bool GetFnPtr(InterpState &S, CodePtr OpPC, const Function *Func) {
  assert(Func);
  S.Stk.push<FunctionPointer>(Func);
  return true;
}

/// Just emit a diagnostic. The expression that caused emission of this
/// op is not valid in a constant context.
inline bool Invalid(InterpState &S, CodePtr OpPC) {
  const SourceLocation &Loc = S.Current->getLocation(OpPC);
  S.FFDiag(Loc, diag::note_invalid_subexpr_in_const_expr)
      << S.Current->getRange(OpPC);
  return false;
}

/// Same here, but only for casts.
inline bool InvalidCast(InterpState &S, CodePtr OpPC, CastKind Kind) {
  const SourceLocation &Loc = S.Current->getLocation(OpPC);

  // FIXME: Support diagnosing other invalid cast kinds.
  if (Kind == CastKind::Reinterpret)
    S.FFDiag(Loc, diag::note_constexpr_invalid_cast)
        << static_cast<unsigned>(Kind) << S.Current->getRange(OpPC);
  return false;
}

inline bool InvalidDeclRef(InterpState &S, CodePtr OpPC,
                           const DeclRefExpr *DR) {
  assert(DR);
  return CheckDeclRef(S, OpPC, DR);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
inline bool OffsetOf(InterpState &S, CodePtr OpPC, const OffsetOfExpr *E) {
  llvm::SmallVector<int64_t> ArrayIndices;
  for (size_t I = 0; I != E->getNumExpressions(); ++I)
    ArrayIndices.emplace_back(S.Stk.pop<int64_t>());

  int64_t Result;
  if (!InterpretOffsetOf(S, OpPC, E, ArrayIndices, Result))
    return false;

  S.Stk.push<T>(T::from(Result));

  return true;
}

//===----------------------------------------------------------------------===//
// Read opcode arguments
//===----------------------------------------------------------------------===//

template <typename T> inline T ReadArg(InterpState &S, CodePtr &OpPC) {
  if constexpr (std::is_pointer<T>::value) {
    uint32_t ID = OpPC.read<uint32_t>();
    return reinterpret_cast<T>(S.P.getNativePointer(ID));
  } else {
    return OpPC.read<T>();
  }
}

template <> inline Floating ReadArg<Floating>(InterpState &S, CodePtr &OpPC) {
  Floating F = Floating::deserialize(*OpPC);
  OpPC += align(F.bytesToSerialize());
  return F;
}

template <>
inline IntegralAP<false> ReadArg<IntegralAP<false>>(InterpState &S,
                                                    CodePtr &OpPC) {
  IntegralAP<false> I = IntegralAP<false>::deserialize(*OpPC);
  OpPC += align(I.bytesToSerialize());
  return I;
}

template <>
inline IntegralAP<true> ReadArg<IntegralAP<true>>(InterpState &S,
                                                  CodePtr &OpPC) {
  IntegralAP<true> I = IntegralAP<true>::deserialize(*OpPC);
  OpPC += align(I.bytesToSerialize());
  return I;
}

} // namespace interp
} // namespace clang

#endif
