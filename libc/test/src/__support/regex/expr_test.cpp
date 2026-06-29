//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for regex AST expressions and pool.
///
//===----------------------------------------------------------------------===//
#include "hdr/regex_macros.h"
#include "src/__support/regex/regex_ast.h"
#include "src/__support/regex/regex_expr_pool.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::regex::Expr;
using LIBC_NAMESPACE::regex::ExprKind;
using LIBC_NAMESPACE::regex::ExprPool;

TEST(LlvmLibcRegexExprTest, Interning) {
  ExprPool pool;
  auto lit_a = pool.make_lit('a');
  ASSERT_TRUE(lit_a.has_value());
  auto lit_a_2 = pool.make_lit('a');
  ASSERT_TRUE(lit_a_2.has_value());
  // Hash-consing: same literal should return the same pointer.
  EXPECT_EQ(lit_a.value(), lit_a_2.value());

  auto lit_b = pool.make_lit('b');
  ASSERT_TRUE(lit_b.has_value());
  EXPECT_NE(lit_a.value(), lit_b.value());

  auto empty_set_1 = pool.empty_set();
  auto empty_set_2 = pool.empty_set();
  EXPECT_EQ(empty_set_1.value(), empty_set_2.value());

  auto empty_str_1 = pool.empty_str();
  auto empty_str_2 = pool.empty_str();
  EXPECT_EQ(empty_str_1.value(), empty_str_2.value());
}

TEST(LlvmLibcRegexExprTest, RecursiveInterning) {
  ExprPool pool;
  auto lit_a = pool.make_lit('a').value();
  auto lit_b = pool.make_lit('b').value();
  auto lit_c = pool.make_lit('c').value();

  // Create (a · b) | c
  auto concat1 = pool.make_concat(lit_a, lit_b).value();
  auto alt1 = pool.make_alt(concat1, lit_c).value();

  // Create (a · b) | c again
  auto concat2 = pool.make_concat(lit_a, lit_b).value();
  auto alt2 = pool.make_alt(concat2, lit_c).value();

  EXPECT_EQ(alt1, alt2);
  EXPECT_EQ(concat1, concat2);
}

TEST(LlvmLibcRegexExprTest, AlgebraicIdentitiesConcat) {
  ExprPool pool;
  auto lit_a = pool.make_lit('a').value();
  auto empty_set = pool.empty_set().value();
  auto empty_str = pool.empty_str().value();

  // (Ø · R) or (R · Ø) => Ø
  EXPECT_EQ(pool.make_concat(empty_set, lit_a).value(), empty_set);
  EXPECT_EQ(pool.make_concat(lit_a, empty_set).value(), empty_set);

  // (ε · R) or (R · ε) => R
  EXPECT_EQ(pool.make_concat(empty_str, lit_a).value(), lit_a);
  EXPECT_EQ(pool.make_concat(lit_a, empty_str).value(), lit_a);
}

TEST(LlvmLibcRegexExprTest, AlgebraicIdentitiesAlt) {
  ExprPool pool;
  auto lit_a = pool.make_lit('a').value();
  auto empty_set = pool.empty_set().value();

  // (Ø | R) or (R | Ø) => R
  EXPECT_EQ(pool.make_alt(empty_set, lit_a).value(), lit_a);
  EXPECT_EQ(pool.make_alt(lit_a, empty_set).value(), lit_a);

  // (R | R) => R (Idempotency)
  EXPECT_EQ(pool.make_alt(lit_a, lit_a).value(), lit_a);
}

TEST(LlvmLibcRegexExprTest, MemoryLimits) {
  ExprPool pool;
  // Verify that the pool correctly enforces MAX_NODE_LIMIT (10000 nodes).
  // We exceed the limit by creating a deep chain of unique concatenations.
  auto lit_a = pool.make_lit('a').value();
  auto current = lit_a;
  for (size_t i = 0; i < 9999; ++i) {
    auto next = pool.make_concat(lit_a, current);
    ASSERT_TRUE(next.has_value());
    current = next.value();
  }

  // Next one should fail.
  auto fail = pool.make_concat(lit_a, current);
  ASSERT_FALSE(fail.has_value());
  EXPECT_EQ(fail.error(), REG_ESPACE);
}
