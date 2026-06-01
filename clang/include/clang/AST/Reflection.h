//===--- Reflection.h - Kind of reflection operands ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the kinds of reflection operands.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_REFLECTION_H
#define LLVM_CLANG_AST_REFLECTION_H

#include "llvm/Support/raw_ostream.h"

namespace clang {

// TODO(Reflection): Add support for Template, Namespace and DeclRefExpr.
enum class ReflectionKind { Null, Type };

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     ReflectionKind Kind) {
  switch (Kind) {
  case ReflectionKind::Type:
    OS << "type";
    break;
  case ReflectionKind::Null:
    OS << "null";
    break;
  }

  return OS;
}

} // namespace clang

#endif
