//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compositional ExprPool builder shared across regex unit tests.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC___SUPPORT_REGEX_EXPR_BUILDER_H
#define LLVM_LIBC_TEST_SRC___SUPPORT_REGEX_EXPR_BUILDER_H

#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/regex/regex_ast.h"
#include "src/__support/regex/regex_expr_pool.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {
namespace testing {

// Compositional wrapper around ExprPool that unwraps ErrorOr<ExprId> so regex
// AST and matcher tests read directly as algebra instead of error boilerplate.
class ExprBuilder {
  regex::ExprPool pool;

public:
  static regex::ExprId unwrap(ErrorOr<regex::ExprId> res) {
    EXPECT_TRUE(res.has_value());
    return res.has_value() ? *res : regex::INVALID_EXPR_ID;
  }

  regex::ExprPool &raw() { return pool; }
  regex::Expr get(regex::ExprId id) const { return pool.get(id); }

  regex::ExprId lit(char c) { return unwrap(pool.make_lit(c)); }
  regex::ExprId star(regex::ExprId body) {
    return unwrap(pool.make_star(body));
  }
  regex::ExprId star(char c) { return star(lit(c)); }
  regex::ExprId rep(regex::ExprId body, uint16_t min, uint16_t max) {
    return unwrap(pool.make_repeat(body, min, max));
  }
  regex::ExprId rep(char c, uint16_t min, uint16_t max) {
    return rep(lit(c), min, max);
  }
  regex::ExprId cat(regex::ExprId l, regex::ExprId r) {
    return unwrap(pool.make_concat(l, r));
  }
  regex::ExprId alt(regex::ExprId l, regex::ExprId r) {
    return unwrap(pool.make_alt(l, r));
  }

  template <typename... Rest>
  regex::ExprId alt(regex::ExprId first, regex::ExprId second, Rest... rest) {
    return alt(alt(first, second), rest...);
  }

  // Returns the idx-th distinct two-character literal pair ("a!", "a\"", ...)
  // for building wide Alt spines without Repeat nodes.
  regex::ExprId distinct_branch(size_t idx) {
    return cat(lit(static_cast<char>('a' + (idx / 60))),
               lit(static_cast<char>('!' + (idx % 60))));
  }

  // Walks a right-associated Alt spine, verifies right-associativity and
  // strictly ascending ExprId order, and returns the total branch count.
  size_t verify_alt_spine(regex::ExprId spine) const {
    if (spine == regex::INVALID_EXPR_ID)
      return 0;
    size_t count = 0;
    regex::ExprId prev = regex::INVALID_EXPR_ID;
    regex::ExprId walk = spine;
    while (walk != regex::EMPTY_SET_ID) {
      const regex::Expr we = pool.get(walk);
      const bool is_alt = (we.kind == regex::ExprKind::Alt);
      const regex::ExprId head = is_alt ? we.bin.left : walk;
      EXPECT_NE(pool.get(head).kind, regex::ExprKind::Alt);
      if (prev != regex::INVALID_EXPR_ID)
        EXPECT_LT(prev, head);
      prev = head;
      ++count;
      walk = is_alt ? we.bin.right : regex::EMPTY_SET_ID;
    }
    return count;
  }

  // Counts occurrences of `target` as a top-level branch of an Alt spine.
  size_t count_branch(regex::ExprId spine, regex::ExprId target) const {
    if (spine == regex::INVALID_EXPR_ID)
      return 0;
    size_t hits = 0;
    regex::ExprId walk = spine;
    while (walk != regex::EMPTY_SET_ID) {
      const regex::Expr we = pool.get(walk);
      const bool is_alt = (we.kind == regex::ExprKind::Alt);
      if ((is_alt ? we.bin.left : walk) == target)
        ++hits;
      walk = is_alt ? we.bin.right : regex::EMPTY_SET_ID;
    }
    return hits;
  }

  // Builds a right-associated literal chain cycling `pattern` (`count`
  // literals) followed by `tail`.  A 3-character cycle ("abc") is asymmetric
  // at both DEPTH and DEPTH - 1, so reversing rebuild order fails immediately.
  template <size_t N>
  regex::ExprId cycle_literals(const char (&pattern)[N], size_t count,
                               regex::ExprId tail = regex::EMPTY_STR_ID) {
    static_assert(N > 1, "pattern must contain at least one character");
    constexpr size_t PERIOD = N - 1;
    regex::ExprId acc = tail;
    for (size_t i = count; i > 0 && acc != regex::INVALID_EXPR_ID; --i)
      acc = cat(lit(pattern[(i - 1) % PERIOD]), acc);
    return acc;
  }
};

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_TEST_SRC___SUPPORT_REGEX_EXPR_BUILDER_H
