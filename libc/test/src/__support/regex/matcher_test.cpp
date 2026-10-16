//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for Brzozowski derivative evaluation and substring matching.
///
//===----------------------------------------------------------------------===//

#include "hdr/regex_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/error_or.h"
#include "src/__support/regex/regex_ast.h"
#include "src/__support/regex/regex_expr_pool.h"
#include "src/__support/regex/regex_matcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::ErrorOr;
using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::regex::derivative;
using LIBC_NAMESPACE::regex::EMPTY_SET_ID;
using LIBC_NAMESPACE::regex::EMPTY_STR_ID;
using LIBC_NAMESPACE::regex::ExprId;
using LIBC_NAMESPACE::regex::ExprPool;
using LIBC_NAMESPACE::regex::match;
using LIBC_NAMESPACE::regex::MatchResult;

namespace {

ErrorOr<ExprId> make_literal_string(ExprPool &pool, string_view str) {
  ErrorOr<ExprId> cur = pool.empty_str();
  if (!cur.has_value())
    return cur;
  for (char c : str) {
    ErrorOr<ExprId> lit = pool.make_lit(c);
    if (!lit.has_value())
      return lit;
    cur = pool.make_concat(*cur, *lit);
    if (!cur.has_value())
      return cur;
  }
  return cur;
}

} // namespace

TEST(LlvmLibcRegexMatcherTest, DerivativeBaseCases) {
  ExprPool pool;
  ErrorOr<ExprId> empty_set = pool.empty_set();
  ErrorOr<ExprId> empty_str = pool.empty_str();
  ErrorOr<ExprId> lit_a = pool.make_lit('a');
  ASSERT_TRUE(empty_set.has_value());
  ASSERT_TRUE(empty_str.has_value());
  ASSERT_TRUE(lit_a.has_value());

  EXPECT_EQ(*derivative(*empty_set, 'a', pool), EMPTY_SET_ID);
  EXPECT_EQ(*derivative(*empty_str, 'a', pool), EMPTY_SET_ID);
  EXPECT_EQ(*derivative(*lit_a, 'a', pool), EMPTY_STR_ID);
  EXPECT_EQ(*derivative(*lit_a, 'b', pool), EMPTY_SET_ID);
}

TEST(LlvmLibcRegexMatcherTest, DerivativeConcatAndAltRules) {
  ExprPool pool;
  const ExprId lit_a = *pool.make_lit('a');
  const ExprId lit_b = *pool.make_lit('b');
  const ExprId empty_str = *pool.empty_str();

  // Non-nullable left: D_a(a . b) = b, D_b(a . b) = EmptySet.
  const ExprId ab = *pool.make_concat(lit_a, lit_b);
  EXPECT_EQ(*derivative(ab, 'a', pool), lit_b);
  EXPECT_EQ(*derivative(ab, 'b', pool), EMPTY_SET_ID);

  // Nullable left: D_b((a | EmptyStr) . b) = EmptyStr.
  const ExprId opt_a = *pool.make_alt(lit_a, empty_str);
  const ExprId opt_a_b = *pool.make_concat(opt_a, lit_b);
  EXPECT_EQ(*derivative(opt_a_b, 'a', pool), lit_b);
  EXPECT_EQ(*derivative(opt_a_b, 'b', pool), EMPTY_STR_ID);

  // Sum rule: D_a(a | b) = EmptyStr, D_b(a | b) = EmptyStr, D_c(a | b) =
  // EmptySet.
  const ExprId a_or_b = *pool.make_alt(lit_a, lit_b);
  EXPECT_EQ(*derivative(a_or_b, 'a', pool), EMPTY_STR_ID);
  EXPECT_EQ(*derivative(a_or_b, 'b', pool), EMPTY_STR_ID);
  EXPECT_EQ(*derivative(a_or_b, 'c', pool), EMPTY_SET_ID);
}

TEST(LlvmLibcRegexMatcherTest, MatchSubstringAndMemoization) {
  ExprPool pool;
  ErrorOr<ExprId> hello = make_literal_string(pool, "hello");
  ASSERT_TRUE(hello.has_value());
  EXPECT_EQ(pool.size(), size_t(10));
  EXPECT_LE(pool.allocated_bytes(), size_t(1024));

  EXPECT_EQ(match(pool, *hello, "say hello world"), MatchResult::Match);
  EXPECT_LE(pool.allocated_bytes(), size_t(1024));

  // Once the derivative states for "say hello" are interned in the pool, a
  // subsequent match over the same input must allocate zero new nodes or bytes.
  const size_t warm_nodes = pool.size();
  const size_t warm_bytes = pool.allocated_bytes();
  EXPECT_EQ(match(pool, *hello, "say hello world"), MatchResult::Match);
  EXPECT_EQ(pool.size(), warm_nodes);
  EXPECT_EQ(pool.allocated_bytes(), warm_bytes);

  EXPECT_EQ(match(pool, *hello, "hell"), MatchResult::NoMatch);
}

TEST(LlvmLibcRegexMatcherTest, OverlappingPrefixMatch) {
  ExprPool pool;
  ErrorOr<ExprId> pat = make_literal_string(pool, "ababac");
  ASSERT_TRUE(pat.has_value());

  // At index 4 ('a'), the partial match "ababa" fails on 'b' at index 5 while
  // a second partial match started at index 2 ("aba") advances to "ababac".
  EXPECT_EQ(match(pool, *pat, "abababac"), MatchResult::Match);
  EXPECT_EQ(match(pool, *pat, "abababab"), MatchResult::NoMatch);
}

TEST(LlvmLibcRegexMatcherTest, PoolExhaustionPropagation) {
  ExprPool pool;
  const ExprId empty_str = *pool.empty_str();
  const ExprId lit_a = *pool.make_lit('a');
  const ExprId lit_b = *pool.make_lit('b');
  const ExprId lit_c = *pool.make_lit('c');

  const ExprId ab = *pool.make_concat(lit_a, lit_b);
  const ExprId opt_a = *pool.make_alt(lit_a, empty_str);
  const ExprId opt_ab = *pool.make_concat(opt_a, lit_b);
  // D_a(((a | "") . b) . c) = b . c, which requires interning Concat(b, c).
  const ExprId need_bc = *pool.make_concat(opt_ab, lit_c);
  const ExprId concat_left_fail = *pool.make_concat(need_bc, lit_a);
  const ExprId concat_right_fail = *pool.make_concat(opt_a, need_bc);
  const ExprId alt_right_fail = *pool.make_alt(lit_a, need_bc);
  const ExprId lit_d = *pool.make_lit('d');
  const ExprId alt_left_fail = *pool.make_alt(need_bc, lit_d);

  // Fill the remainder of the pool to MAX_NODE_LIMIT (10000 nodes).
  ExprId chain = lit_a;
  while (pool.size() < 10000)
    chain = *pool.make_concat(lit_a, chain);

  const ExprId exhausted_exprs[] = {need_bc, concat_left_fail,
                                    concat_right_fail, alt_left_fail,
                                    alt_right_fail};
  for (ExprId expr : exhausted_exprs) {
    ErrorOr<ExprId> err = derivative(expr, 'a', pool);
    ASSERT_FALSE(err.has_value());
    EXPECT_EQ(err.error(), REG_ESPACE);
  }

  EXPECT_EQ(match(pool, need_bc, "a"), MatchResult::OutOfMemory);
  EXPECT_EQ(match(pool, ab, "a"), MatchResult::OutOfMemory);
}
