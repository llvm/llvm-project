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
#include "clang/Sema/Sema.h"

namespace clang {

SemaDirectX::SemaDirectX(Sema &S) : SemaBase(S) {}

bool SemaDirectX::CheckDirectXBuiltinFunctionCall(unsigned BuiltinID,
                                                  CallExpr *TheCall) {
  return false;
}
} // namespace clang
