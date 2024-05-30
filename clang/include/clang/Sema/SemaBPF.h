//===----- SemaBPF.h ------- BPF target-specific routines -----*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis functions specific to BPF.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMABPF_H
#define LLVM_CLANG_SEMA_SEMABPF_H

#include "clang/AST/Expr.h"
#include "clang/Sema/SemaBase.h"

namespace clang {
class SemaBPF : public SemaBase {
public:
  SemaBPF(Sema &S);

  bool CheckBPFBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMABPF_H
