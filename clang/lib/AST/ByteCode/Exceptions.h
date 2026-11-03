//===----------------------- Exceptions.h  ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_EXCEPTIONS_H
#define LLVM_CLANG_AST_INTERP_EXCEPTIONS_H

#include "PrimType.h"
#include "clang/Basic/OptionalUnsigned.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {
class Type;
class ASTContext;

namespace interp {
class Block;

struct ExceptionTableEntry {
  unsigned CodeStart;
  unsigned CodeEnd;
  unsigned Target;
  UnsignedOrNone DeclOffset;
  /// If CatchType is nullptr, this is a catch-all handler.
  const Type *CatchType;

  /// Check if this exception table entry can catch an exception thrown of the
  /// given type.
  bool canCatch(const Type *ThrowType, const ASTContext &ASTCtx) const;
};

/// A thrown value.
struct ThrowValue {
  const Type *Ty;
  Block *B;
  SourceLocation Loc;
  OptPrimType T;
  unsigned CastOffset = 0;
  bool Caught = false;

  explicit ThrowValue(const Type *Ty, SourceLocation Loc, Block *B,
                      OptPrimType T)
      : Ty(Ty), B(B), Loc(Loc), T(T) {}
};

} // namespace interp
} // namespace clang
#endif
