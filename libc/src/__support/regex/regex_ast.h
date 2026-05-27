//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// AST nodes for Regular Expressions (Class Definitions).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_REGEX_REGEX_AST_H
#define LLVM_LIBC_SRC___SUPPORT_REGEX_REGEX_AST_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

/// Enumeration of Regular Expression AST node types.
enum class ExprKind {
  /// Represents the empty set (matches nothing).
  EmptySet,
  /// Represents the empty string (matches the empty string).
  EmptyStr,
  /// A literal character match.
  Literal,
  /// Concatenation of two expressions (left followed by right).
  Concat,
  /// Alternation between two expressions (left or right).
  Alt,
};

/// A node in the Regular Expression Abstract Syntax Tree.
///
/// Expressions are represented as a hash-consed DAG to enable efficient
/// derivative-based matching. This structure is intended to be managed by
/// an ExprPool.
struct Expr {
  /// The type of this expression node.
  ExprKind kind;
  union {
    /// Character value for Literal nodes.
    char ch;
    /// Sub-expressions for Concat and Alt nodes.
    struct {
      Expr *left;
      Expr *right;
    } bin;
  };

  /// Default constructor creates an EmptySet node.
  constexpr Expr() : kind(ExprKind::EmptySet), ch('\0') {}
  /// Create a node of a specific kind with no data.
  constexpr Expr(ExprKind k) : kind(k), ch('\0') {}
  /// Create a Literal node.
  constexpr Expr(char c) : kind(ExprKind::Literal), ch(c) {}
  /// Create a binary node (Concat or Alt).
  constexpr Expr(ExprKind k, Expr *l, Expr *r) : kind(k), bin{l, r} {}

  /// Equivalence check for hash-consing.
  bool operator==(const Expr &other) const {
    if (kind != other.kind)
      return false;
    switch (kind) {
    case ExprKind::EmptySet:
    case ExprKind::EmptyStr:
      return true;
    case ExprKind::Literal:
      return ch == other.ch;
    case ExprKind::Concat:
    case ExprKind::Alt:
      return bin.left == other.bin.left && bin.right == other.bin.right;
    }
    return false;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_REGEX_REGEX_AST_H
