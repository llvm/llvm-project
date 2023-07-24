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
#include "clang/Basic/Builtins.h"

namespace clang {
namespace interp {

template <typename T> T getParam(InterpFrame *Frame, unsigned Index) {
  unsigned Offset = Frame->getFunction()->getParamOffset(Index);
  return Frame->getParam<T>(Offset);
}

static bool interp__builtin_strcmp(InterpState &S, CodePtr OpPC,
                                   InterpFrame *Frame) {
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

  S.Stk.push<Integral<32, true>>(Integral<32, true>::from(Result));
  return true;
}

bool InterpretBuiltin(InterpState &S, CodePtr OpPC, const Function *F) {
  InterpFrame *Frame = S.Current;
  APValue Dummy;

  switch (F->getBuiltinID()) {
  case Builtin::BI__builtin_is_constant_evaluated:
    S.Stk.push<Boolean>(Boolean::from(S.inConstantContext()));
    return Ret<PT_Bool, true>(S, OpPC, Dummy);
  case Builtin::BI__builtin_assume:
    return RetVoid<true>(S, OpPC, Dummy);
  case Builtin::BI__builtin_strcmp:
    if (interp__builtin_strcmp(S, OpPC, Frame))
      return Ret<PT_Sint32, true>(S, OpPC, Dummy);
    return false;
  default:
    return false;
  }

  return false;
}

} // namespace interp
} // namespace clang
