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

/// This is a slightly simplified version of the Ret() we have in Interp.cpp
/// If they end up diverging in the future, we should get rid of the code
/// duplication.
template <PrimType Name, class T = typename PrimConv<Name>::T>
static bool Ret(InterpState &S, CodePtr &PC) {
  S.CallStackDepth--;
  const T &Ret = S.Stk.pop<T>();

  assert(S.Current->getFrameOffset() == S.Stk.size() && "Invalid frame");
  if (!S.checkingPotentialConstantExpression())
    S.Current->popArgs();

  InterpFrame *Caller = S.Current->Caller;
  assert(Caller);

  PC = S.Current->getRetPC();
  delete S.Current;
  S.Current = Caller;
  S.Stk.push<T>(Ret);

  return true;
}

bool InterpretBuiltin(InterpState &S, CodePtr PC, unsigned BuiltinID) {
  switch (BuiltinID) {
  case Builtin::BI__builtin_is_constant_evaluated:
    S.Stk.push<Boolean>(Boolean::from(S.inConstantContext()));
    Ret<PT_Bool>(S, PC);
    return true;
  }

  return false;
}

} // namespace interp
} // namespace clang
