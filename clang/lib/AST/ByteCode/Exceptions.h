//===----------------------- Exceptions.h  ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_EXCEPTIONS_H
#define LLVM_CLANG_AST_INTERP_EXCEPTIONS_H

#include "clang/Basic/OptionalUnsigned.h"

namespace clang {
class Type;

namespace interp {

struct ExceptionTableEntry {
  unsigned CodeStart;
  unsigned CodeEnd;
  unsigned Target;
  UnsignedOrNone DeclOffset;
  /// If CatchType is nullptr, this is a catch-all handler.
  const Type *CatchType;

  /// Check if this exception table entry can catch an exception thrown of the
  /// given type.
  bool canCatch(const Type *ThrowType) const;
};

} // namespace interp
} // namespace clang
#endif
