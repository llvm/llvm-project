//===- SemaDirectX.cpp - Semantic Analysis for DirectX constructs----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for DirectX constructs.
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaDirectX.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Sema/Sema.h"

namespace clang {

SemaDirectX::SemaDirectX(Sema &S) : SemaBase(S) {}

bool SemaDirectX::CheckDirectXBuiltinFunctionCall(unsigned BuiltinID,
                                                  CallExpr *TheCall) {
  switch (BuiltinID) {
  case DirectX::BI__builtin_dx_dot2add: {
  }
  }

  return false;
}
} // namespace clang
