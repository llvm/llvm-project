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

bool InterpretBuiltin(InterpState &S, CodePtr &PC, unsigned BuiltinID) {
  APValue Dummy;

  switch (BuiltinID) {
  case Builtin::BI__builtin_is_constant_evaluated:
    S.Stk.push<Boolean>(Boolean::from(S.inConstantContext()));
    return Ret<PT_Bool, true>(S, PC, Dummy);
  case Builtin::BI__builtin_assume:
    return RetVoid<true>(S, PC, Dummy);
  default:
    return false;
  }

  return false;
}

} // namespace interp
} // namespace clang
