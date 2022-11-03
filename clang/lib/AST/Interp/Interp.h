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

#include "Function.h"
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

using APInt = llvm::APInt;
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
/// Checks if a pointer is null.
bool CheckNull(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
               CheckSubobjectKind CSK);

/// Checks if a pointer is in range.
bool CheckRange(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                AccessKinds AK);

/// Checks if a field from which a pointer is going to be derived is valid.
bool CheckRange(InterpState &S, CodePtr OpPC, const Pointer &Ptr,
                CheckSubobjectKind CSK);

/// Checks if a pointer points to const storage.
bool CheckConst(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a pointer points to a mutable field.
bool CheckMutable(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a value can be loaded from a block.
bool CheckLoad(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a value can be stored in a block.
bool CheckStore(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a method can be invoked on an object.
bool CheckInvoke(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a value can be initialized.
bool CheckInit(InterpState &S, CodePtr OpPC, const Pointer &Ptr);

/// Checks if a method can be called.
bool CheckCallable(InterpState &S, CodePtr OpPC, Function *F);

/// Checks the 'this' pointer.
bool CheckThis(InterpState &S, CodePtr OpPC, const Pointer &This);

/// Checks if a method is pure virtual.
bool CheckPure(InterpState &S, CodePtr OpPC, const CXXMethodDecl *MD);

/// Checks if Div/Rem operation on LHS and RHS is valid.
template <typename T>
bool CheckDivRem(InterpState &S, CodePtr OpPC, const T &LHS, const T &RHS) {
  if (RHS.isZero()) {
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_expr_divide_by_zero);
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

template <typename T> inline bool IsTrue(const T &V) { return !V.isZero(); }

/// Interpreter entry point.
bool Interpret(InterpState &S, APValue &Result);

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
    S.report(Loc, diag::warn_integer_constant_overflow) << Trunc << Type;
    return true;
  } else {
    S.CCEDiag(E, diag::note_constexpr_overflow) << Value << Type;
    return S.noteUndefinedBehavior();
  }
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Add(InterpState &S, CodePtr OpPC) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();
  const unsigned Bits = RHS.bitWidth() + 1;
  return AddSubMulHelper<T, T::add, std::plus>(S, OpPC, Bits, LHS, RHS);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Sub(InterpState &S, CodePtr OpPC) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();
  const unsigned Bits = RHS.bitWidth() + 1;
  return AddSubMulHelper<T, T::sub, std::minus>(S, OpPC, Bits, LHS, RHS);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Mul(InterpState &S, CodePtr OpPC) {
  const T &RHS = S.Stk.pop<T>();
  const T &LHS = S.Stk.pop<T>();
  const unsigned Bits = RHS.bitWidth() * 2;
  return AddSubMulHelper<T, T::mul, std::multiplies>(S, OpPC, Bits, LHS, RHS);
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
  const T &Val = S.Stk.pop<T>();
  T Result;
  T::neg(Val, &Result);

  S.Stk.push<T>(Result);
  return true;
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
  T Value = Ptr.deref<T>();
  T Result;

  if constexpr (DoPush == PushVal::Yes)
    S.Stk.push<T>(Result);

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
    S.report(Loc, diag::warn_integer_constant_overflow) << Trunc << Type;
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
  // FIXME: Check initialization of Ptr
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  return IncDecHelper<T, IncDecOp::Inc, PushVal::Yes>(S, OpPC, Ptr);
}

/// 1) Pops a pointer from the stack
/// 2) Load the value from the pointer
/// 3) Writes the value increased by one back to the pointer
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool IncPop(InterpState &S, CodePtr OpPC) {
  // FIXME: Check initialization of Ptr
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  return IncDecHelper<T, IncDecOp::Inc, PushVal::No>(S, OpPC, Ptr);
}

/// 1) Pops a pointer from the stack
/// 2) Load the value from the pointer
/// 3) Writes the value decreased by one back to the pointer
/// 4) Pushes the original (pre-dec) value on the stack.
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Dec(InterpState &S, CodePtr OpPC) {
  // FIXME: Check initialization of Ptr
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  return IncDecHelper<T, IncDecOp::Dec, PushVal::Yes>(S, OpPC, Ptr);
}

/// 1) Pops a pointer from the stack
/// 2) Load the value from the pointer
/// 3) Writes the value decreased by one back to the pointer
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool DecPop(InterpState &S, CodePtr OpPC) {
  // FIXME: Check initialization of Ptr
  const Pointer &Ptr = S.Stk.pop<Pointer>();

  return IncDecHelper<T, IncDecOp::Dec, PushVal::No>(S, OpPC, Ptr);
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

template <>
inline bool CmpHelper<Pointer>(InterpState &S, CodePtr OpPC, CompareFn Fn) {
  using BoolT = PrimConv<PT_Bool>::T;
  const Pointer &RHS = S.Stk.pop<Pointer>();
  const Pointer &LHS = S.Stk.pop<Pointer>();

  if (!Pointer::hasSameBase(LHS, RHS)) {
    const SourceInfo &Loc = S.Current->getSource(OpPC);
    S.FFDiag(Loc, diag::note_invalid_subexpr_in_const_expr);
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
  S.Stk.push<T>(S.Current->getLocal<T>(I));
  return true;
}

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
  auto *B = S.P.getGlobal(I);
  if (B->isExtern())
    return false;
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
  S.P.getGlobal(I)->deref<T>() = S.Stk.pop<T>();
  return true;
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

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitThisBitField(InterpState &S, CodePtr OpPC, const Record::Field *F) {
  if (S.checkingPotentialConstantExpression())
    return false;
  const Pointer &This = S.Current->getThis();
  if (!CheckThis(S, OpPC, This))
    return false;
  const Pointer &Field = This.atField(F->Offset);
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
/// 2) Pops a pointer from the stack
/// 3) Pushes the value to field I of the pointer on the stack
template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitField(InterpState &S, CodePtr OpPC, uint32_t I) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Field = S.Stk.pop<Pointer>().atField(I);
  Field.deref<T>() = Value;
  Field.activate();
  Field.initialize();
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool InitBitField(InterpState &S, CodePtr OpPC, const Record::Field *F) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Field = S.Stk.pop<Pointer>().atField(F->Offset);
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
  if (!CheckNull(S, OpPC, Ptr, CSK_Field))
    return false;
  if (!CheckExtern(S, OpPC, Ptr))
    return false;
  if (!CheckRange(S, OpPC, Ptr, CSK_Field))
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

inline bool GetPtrBase(InterpState &S, CodePtr OpPC, uint32_t Off) {
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckNull(S, OpPC, Ptr, CSK_Base))
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
  Ptr.deref<T>() = Value;
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool StorePop(InterpState &S, CodePtr OpPC) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckStore(S, OpPC, Ptr))
    return false;
  Ptr.deref<T>() = Value;
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool StoreBitField(InterpState &S, CodePtr OpPC) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.peek<Pointer>();
  if (!CheckStore(S, OpPC, Ptr))
    return false;
  if (auto *FD = Ptr.getField()) {
    Ptr.deref<T>() = Value.truncate(FD->getBitWidthValue(S.getCtx()));
  } else {
    Ptr.deref<T>() = Value;
  }
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool StoreBitFieldPop(InterpState &S, CodePtr OpPC) {
  const T &Value = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckStore(S, OpPC, Ptr))
    return false;
  if (auto *FD = Ptr.getField()) {
    Ptr.deref<T>() = Value.truncate(FD->getBitWidthValue(S.getCtx()));
  } else {
    Ptr.deref<T>() = Value;
  }
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

template <class T, bool Add> bool OffsetHelper(InterpState &S, CodePtr OpPC) {
  // Fetch the pointer and the offset.
  const T &Offset = S.Stk.pop<T>();
  const Pointer &Ptr = S.Stk.pop<Pointer>();
  if (!CheckNull(S, OpPC, Ptr, CSK_ArrayIndex))
    return false;
  if (!CheckRange(S, OpPC, Ptr, CSK_ArrayToPointer))
    return false;

  // Get a version of the index comparable to the type.
  T Index = T::from(Ptr.getIndex(), Offset.bitWidth());
  // A zero offset does not change the pointer, but in the case of an array
  // it has to be adjusted to point to the first element instead of the array.
  if (Offset.isZero()) {
    S.Stk.push<Pointer>(Index.isZero() ? Ptr.atIndex(0) : Ptr);
    return true;
  }
  // Arrays of unknown bounds cannot have pointers into them.
  if (!CheckArray(S, OpPC, Ptr))
    return false;

  // Compute the largest index into the array.
  unsigned MaxIndex = Ptr.getNumElems();

  // Helper to report an invalid offset, computed as APSInt.
  auto InvalidOffset = [&]() {
    const unsigned Bits = Offset.bitWidth();
    APSInt APOffset(Offset.toAPSInt().extend(Bits + 2), false);
    APSInt APIndex(Index.toAPSInt().extend(Bits + 2), false);
    APSInt NewIndex = Add ? (APIndex + APOffset) : (APIndex - APOffset);
    S.CCEDiag(S.Current->getSource(OpPC), diag::note_constexpr_array_index)
        << NewIndex
        << /*array*/ static_cast<int>(!Ptr.inArray())
        << static_cast<unsigned>(MaxIndex);
    return false;
  };

  unsigned MaxOffset = MaxIndex - Ptr.getIndex();
  if constexpr (Add) {
    // If the new offset would be negative, bail out.
    if (Offset.isNegative() && (Offset.isMin() || -Offset > Index))
      return InvalidOffset();

    // If the new offset would be out of bounds, bail out.
    if (Offset.isPositive() && Offset > MaxOffset)
      return InvalidOffset();
  } else {
    // If the new offset would be negative, bail out.
    if (Offset.isPositive() && Index < Offset)
      return InvalidOffset();

    // If the new offset would be out of bounds, bail out.
    if (Offset.isNegative() && (Offset.isMin() || -Offset > MaxOffset))
      return InvalidOffset();
  }

  // Offset is valid - compute it on unsigned.
  int64_t WideIndex = static_cast<int64_t>(Index);
  int64_t WideOffset = static_cast<int64_t>(Offset);
  int64_t Result = Add ? (WideIndex + WideOffset) : (WideIndex - WideOffset);
  S.Stk.push<Pointer>(Ptr.atIndex(static_cast<unsigned>(Result)));
  return true;
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool AddOffset(InterpState &S, CodePtr OpPC) {
  return OffsetHelper<T, true>(S, OpPC);
}

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool SubOffset(InterpState &S, CodePtr OpPC) {
  return OffsetHelper<T, false>(S, OpPC);
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

//===----------------------------------------------------------------------===//
// Zero, Nullptr
//===----------------------------------------------------------------------===//

template <PrimType Name, class T = typename PrimConv<Name>::T>
bool Zero(InterpState &S, CodePtr OpPC) {
  S.Stk.push<T>(T::zero());
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

  unsigned URHS = static_cast<unsigned>(RHS);
  S.Stk.push<LT>(LT::from(static_cast<unsigned>(LHS) >> URHS, LHS.bitWidth()));
  return true;
}

template <PrimType NameL, PrimType NameR>
inline bool Shl(InterpState &S, CodePtr OpPC) {
  using LT = typename PrimConv<NameL>::T;
  using RT = typename PrimConv<NameR>::T;
  const auto &RHS = S.Stk.pop<RT>();
  const auto &LHS = S.Stk.pop<LT>();
  const unsigned Bits = LHS.bitWidth();

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

  unsigned URHS = static_cast<unsigned>(RHS);
  S.Stk.push<LT>(LT::from(static_cast<unsigned>(LHS) << URHS, LHS.bitWidth()));

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

inline bool Call(InterpState &S, CodePtr &PC, const Function *Func) {
  auto NewFrame = std::make_unique<InterpFrame>(S, Func, PC);
  if (Func->hasThisPointer()) {
    if (!CheckInvoke(S, PC, NewFrame->getThis())) {
      return false;
    }
    // TODO: CheckCallable
  }

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

} // namespace interp
} // namespace clang

#endif
