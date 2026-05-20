//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Pool for Regular Expression AST nodes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_REGEX_REGEX_EXPR_POOL_H
#define LLVM_LIBC_SRC___SUPPORT_REGEX_REGEX_EXPR_POOL_H

#include "src/__support/CPP/expected.h"
#include "src/__support/macros/config.h"
#include "src/__support/regex/regex_ast.h"
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

/// An arena-based pool for Regular Expression AST nodes.
///
/// This class manages the allocation and hash-consing of Expr nodes. All
/// nodes created through this pool are owned by it and will be freed when
/// the pool is destroyed. Hash-consing ensures that identical expressions
/// are represented by the same pointer, enabling fast comparison and
/// derivative normalization.
class ExprPool {
  /// Internal storage block for AST nodes.
  ///
  /// Blocks are allocated on demand to avoid large contiguous allocations
  /// and are linked together in a list for cleanup.
  /// TODO: Consider adopting cpp::forward_list for block management once
  /// it is available in LLVM-libc.
  struct Block {
    /// Number of Expr nodes stored in each block.
    static constexpr size_t SIZE = 256;
    /// The actual storage for Expr nodes.
    Expr nodes[SIZE];
    /// Pointer to the next block in the chain.
    Block *next = nullptr;
    /// Number of nodes currently used in this block.
    size_t used = 0;

    /// Initializes an empty block.
    Block();
  };

  /// The first block in the allocation chain.
  Block *head = nullptr;
  /// The block currently being used for new node allocations.
  Block *current = nullptr;
  /// Total number of nodes allocated across all blocks.
  size_t node_count = 0;

  /// Size of the hash table used for hash-consing.
  /// Must be significantly larger than MAX_NODES to maintain efficiency.
  static constexpr size_t HASH_SIZE = 16384;
  /// Hash table storing pointers to unique Expr nodes.
  Expr *hashtable[HASH_SIZE];

  /// Core hash-consing function (Interning).
  ///
  /// Guarantees that for any two identical structural definitions of an Expr,
  /// this function will return the same pointer. This enables O(1) structural
  /// equality via pointer comparison.
  ///
  /// \param e A structural definition (proto-node) to intern.
  /// \returns A pointer to the unique, stable instance in the arena,
  ///          or REG_ESPACE on failure.
  cpp::expected<Expr *, int> intern(const Expr &e);

  /// Maximum number of nodes allowed in the pool to prevent memory exhaustion.
  static constexpr size_t MAX_NODES = 10000;

public:
  ExprPool();
  ~ExprPool();

  // TODO: Use fluent interface (and_then, transform) for these factories once
  // implemented in cpp::expected.

  /// Returns an EmptySet node.
  cpp::expected<Expr *, int> empty_set();
  /// Returns an EmptyStr node.
  cpp::expected<Expr *, int> empty_str();
  /// Creates or returns an existing Literal node for the given character.
  cpp::expected<Expr *, int> make_lit(char c);
  /// Normalizing factory for Concatenation (L · R).
  ///
  /// Applies algebraic simplifications before interning:
  /// - (Ø · R) or (R · Ø) => Ø
  /// - (ε · R) or (R · ε) => R
  cpp::expected<Expr *, int> make_concat(Expr *l, Expr *r);

  /// Normalizing factory for Alternation (L | R).
  ///
  /// Applies algebraic simplifications before interning:
  /// - (Ø | R) or (R | Ø) => R
  /// - (R | R) => R (Idempotency)
  cpp::expected<Expr *, int> make_alt(Expr *l, Expr *r);
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_REGEX_REGEX_EXPR_POOL_H
