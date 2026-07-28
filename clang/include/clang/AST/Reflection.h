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

#include "clang/AST/TypeBase.h"
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

/// [expr.reflect] p5, if a reflect-expression R matches the form
/// ^^reflection-name it is interpreted as such; the identifier is looked up
/// and the representation of R is determined as follows:
/// - if lookup finds a type alias A, R represents the type the underlying
///   entity of A if A was introduced by the declaration of a template
///   parameter; otherwise, R represents A.

/// [expr.reflect] p6, Given reflect-expression R of the form ^^type-id,
/// if type-id is neither a placeholder type nor
/// in the form of nested-name-specifier_opt template_opt simple-template-id
/// then R represents the type denoted by the type-id

// In particular, this means that e.g. '^^const Alias' is reflection of
// a type, not an alias. For example:
//
// using foo = const int;
// ^^int       // Type
// ^^const int // Type
// ^^foo       // Alias
// ^^const foo // Type
inline bool isTypeAliasAsReflectionName(QualType QT) {
  return QT.getLocalQualifiers() == Qualifiers{};
}

} // namespace clang

#endif
