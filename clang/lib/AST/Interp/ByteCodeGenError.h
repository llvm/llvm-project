//===--- ByteCodeGenError.h - Byte code generation error ----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_BYTECODEGENERROR_H
#define LLVM_CLANG_AST_INTERP_BYTECODEGENERROR_H

#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace interp {

/// Error thrown by the compiler.
struct ByteCodeGenError : public llvm::ErrorInfo<ByteCodeGenError> {
public:
  ByteCodeGenError(SourceRange Range) : Range(Range) {}
  ByteCodeGenError(const Stmt *S) : ByteCodeGenError(S->getSourceRange()) {}
  ByteCodeGenError(const Decl *D) : ByteCodeGenError(D->getSourceRange()) {}

  void log(raw_ostream &OS) const override { OS << "unimplemented feature"; }

  const SourceRange &getRange() const { return Range; }

  static char ID;

private:
  // Range of the item where the error occurred.
  SourceRange Range;

  // Users are not expected to use error_code.
  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

} // namespace interp
} // namespace clang

#endif
