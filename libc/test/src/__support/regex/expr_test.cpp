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

using LIBC_NAMESPACE::regex::ANCHOR_BOL;
using LIBC_NAMESPACE::regex::ANCHOR_EOL;
using LIBC_NAMESPACE::regex::ByteSet;
using LIBC_NAMESPACE::regex::EMPTY_SET_ID;
using LIBC_NAMESPACE::regex::Expr;
using LIBC_NAMESPACE::regex::ExprId;
using LIBC_NAMESPACE::regex::ExprKind;
using LIBC_NAMESPACE::regex::ExprPool;
using LIBC_NAMESPACE::regex::INVALID_EXPR_ID;

TEST(LlvmLibcRegexExprTest, Interning) {
  ExprPool pool;
  EXPECT_EQ(sizeof(Expr), size_t(16));
  EXPECT_EQ(pool.allocated_bytes(), size_t(0));
  auto lit_a = pool.make_lit('a');
  ASSERT_TRUE(lit_a.has_value());
  auto lit_a_2 = pool.make_lit('a');
  ASSERT_TRUE(lit_a_2.has_value());
  // Hash-consing: same literal should return the same ExprId.
  EXPECT_EQ(*lit_a, *lit_a_2);
  EXPECT_FALSE(pool.has_anchors(*lit_a));

  auto lit_b = pool.make_lit('b');
  ASSERT_TRUE(lit_b.has_value());
  EXPECT_NE(*lit_a, *lit_b);

  auto bol_1 = pool.make_anchor(ANCHOR_BOL);
  auto bol_2 = pool.make_anchor(ANCHOR_BOL);
  auto eol = pool.make_anchor(ANCHOR_EOL);
  ASSERT_TRUE(bol_1.has_value());
  ASSERT_TRUE(bol_2.has_value());
  ASSERT_TRUE(eol.has_value());
  EXPECT_EQ(*bol_1, *bol_2);
  EXPECT_NE(*bol_1, *eol);
  EXPECT_FALSE(pool.is_nullable(*bol_1));
  EXPECT_TRUE(pool.has_anchors(*bol_1));
  EXPECT_TRUE(pool.has_anchors(*pool.make_concat(*bol_1, *lit_a)));
  EXPECT_TRUE(pool.has_anchors(*pool.make_alt(*lit_a, *eol)));
  EXPECT_FALSE(pool.has_anchors(*pool.make_concat(*lit_a, *lit_b)));

  auto empty_set_1 = pool.empty_set();
  auto empty_set_2 = pool.empty_set();
  ASSERT_TRUE(empty_set_1.has_value());
  ASSERT_TRUE(empty_set_2.has_value());
  EXPECT_EQ(*empty_set_1, *empty_set_2);
  EXPECT_FALSE(pool.is_nullable(*empty_set_1));

  auto empty_str_1 = pool.empty_str();
  auto empty_str_2 = pool.empty_str();
  ASSERT_TRUE(empty_str_1.has_value());
  ASSERT_TRUE(empty_str_2.has_value());
  EXPECT_EQ(*empty_str_1, *empty_str_2);
  EXPECT_TRUE(pool.is_nullable(*empty_str_1));
  EXPECT_LE(pool.allocated_bytes(), size_t(1024));
}

TEST(LlvmLibcRegexExprTest, RecursiveInterning) {
  ExprPool pool;
  const ExprId lit_a = *pool.make_lit('a');
  const ExprId lit_b = *pool.make_lit('b');
  const ExprId lit_c = *pool.make_lit('c');

  // Create (a · b) | c
  const ExprId concat1 = *pool.make_concat(lit_a, lit_b);
  const ExprId alt1 = *pool.make_alt(concat1, lit_c);

  // Create (a · b) | c again
  const ExprId concat2 = *pool.make_concat(lit_a, lit_b);
  const ExprId alt2 = *pool.make_alt(concat2, lit_c);

  EXPECT_EQ(alt1, alt2);
  EXPECT_EQ(concat1, concat2);
}

TEST(LlvmLibcRegexExprTest, ByteSetNormalizationAndDeduplication) {
  ExprPool pool;
  // 0-bit and 1-bit ByteSets simplify to EmptySet and Literal without
  // allocating a ByteSet slot.
  ByteSet bs;
  EXPECT_EQ(*pool.make_byte_set(bs), EMPTY_SET_ID);
  bs.set(static_cast<uint8_t>('x'));
  EXPECT_EQ(*pool.make_byte_set(bs), *pool.make_lit('x'));
  EXPECT_EQ(pool.num_byte_sets(), size_t(0));

  // Multi-bit ByteSet allocates and deduplicates across identical sets.
  bs.set(static_cast<uint8_t>('a'));
  const ExprId bs1 = *pool.make_byte_set(bs);
  const ExprId bs2 = *pool.make_byte_set(bs);
  EXPECT_EQ(bs1, bs2);
  EXPECT_EQ(pool.num_byte_sets(), size_t(1));
  EXPECT_TRUE(pool.get_byte_set(0) == bs);
  EXPECT_LE(pool.allocated_bytes(), size_t(1024));

  // Grow past INITIAL_BYTE_SETS_CAPACITY (4).
  for (uint8_t i = 0; i < 5; ++i) {
    ByteSet multi;
    multi.set(i);
    multi.set(static_cast<uint8_t>(i + 10));
    ASSERT_TRUE(pool.make_byte_set(multi).has_value());
  }
  EXPECT_EQ(pool.num_byte_sets(), size_t(6));
  EXPECT_TRUE(Expr::make_byte_set(0) == Expr::make_byte_set(0));
  EXPECT_FALSE(Expr::make_byte_set(0) == Expr::make_byte_set(1));
}

TEST(LlvmLibcRegexExprTest, AlgebraicIdentitiesConcat) {
  ExprPool pool;
  const ExprId lit_a = *pool.make_lit('a');
  const ExprId lit_b = *pool.make_lit('b');
  const ExprId empty_set = *pool.empty_set();
  const ExprId empty_str = *pool.empty_str();

  // Invalid IDs return REG_BADPAT.
  auto bad_left = pool.make_concat(INVALID_EXPR_ID, lit_a);
  ASSERT_FALSE(bad_left.has_value());
  EXPECT_EQ(bad_left.error(), REG_BADPAT);
  auto bad_right = pool.make_concat(lit_a, INVALID_EXPR_ID);
  ASSERT_FALSE(bad_right.has_value());
  EXPECT_EQ(bad_right.error(), REG_BADPAT);

  // (Ø · R) or (R · Ø) => Ø
  EXPECT_EQ(*pool.make_concat(empty_set, lit_a), empty_set);
  EXPECT_EQ(*pool.make_concat(lit_a, empty_set), empty_set);

  // (ε · R) or (R · ε) => R
  EXPECT_EQ(*pool.make_concat(empty_str, lit_a), lit_a);
  EXPECT_EQ(*pool.make_concat(lit_a, empty_str), lit_a);

  // Right-associativity: (a · b) · c == a · (b · c).
  const ExprId lit_c = *pool.make_lit('c');
  const ExprId ab = *pool.make_concat(lit_a, lit_b);
  const ExprId bc = *pool.make_concat(lit_b, lit_c);
  EXPECT_EQ(*pool.make_concat(ab, lit_c), *pool.make_concat(lit_a, bc));

  // Nullability of (L · R) requires both L and R to be nullable.
  const ExprId opt_a = *pool.make_alt(lit_a, empty_str);
  const ExprId opt_b = *pool.make_alt(lit_b, empty_str);
  EXPECT_FALSE(pool.is_nullable(*pool.make_concat(lit_a, opt_b)));
  EXPECT_FALSE(pool.is_nullable(*pool.make_concat(opt_a, lit_b)));
  EXPECT_TRUE(pool.is_nullable(*pool.make_concat(opt_a, opt_b)));

  // Expr::operator== structural comparisons across kinds, literals, and binary
  // children.
  EXPECT_FALSE(Expr::make_empty_set() == Expr::make_literal('a'));
  EXPECT_FALSE(Expr::make_literal('a') == Expr::make_literal('b'));
  EXPECT_FALSE(Expr::make_concat(lit_a, lit_b) ==
               Expr::make_concat(lit_a, opt_b));
  EXPECT_FALSE(Expr::make_concat(lit_a, lit_b) ==
               Expr::make_concat(lit_b, lit_b));
}

TEST(LlvmLibcRegexExprTest, AlgebraicIdentitiesAltAndStar) {
  ExprPool pool;
  const ExprId lit_a = *pool.make_lit('a');
  const ExprId lit_b = *pool.make_lit('b');
  const ExprId lit_c = *pool.make_lit('c');
  const ExprId empty_set = *pool.empty_set();
  const ExprId empty_str = *pool.empty_str();

  // Invalid IDs return REG_BADPAT.
  auto bad_left = pool.make_alt(INVALID_EXPR_ID, lit_a);
  ASSERT_FALSE(bad_left.has_value());
  EXPECT_EQ(bad_left.error(), REG_BADPAT);
  auto bad_right = pool.make_alt(lit_a, INVALID_EXPR_ID);
  ASSERT_FALSE(bad_right.has_value());
  EXPECT_EQ(bad_right.error(), REG_BADPAT);

  // (Ø | R) or (R | Ø) => R
  EXPECT_EQ(*pool.make_alt(empty_set, lit_a), lit_a);
  EXPECT_EQ(*pool.make_alt(lit_a, empty_set), lit_a);

  // (R | R) => R (Idempotency)
  EXPECT_EQ(*pool.make_alt(lit_a, lit_a), lit_a);

  // (A | B) == (B | A) (Commutativity)
  const ExprId ab = *pool.make_alt(lit_a, lit_b);
  EXPECT_EQ(ab, *pool.make_alt(lit_b, lit_a));

  // Canonical right-associated sorted deduplication across 3-way alternations.
  const ExprId bc = *pool.make_alt(lit_b, lit_c);
  const ExprId ac = *pool.make_alt(lit_a, lit_c);
  const ExprId abc = *pool.make_alt(lit_a, bc);
  EXPECT_EQ(*pool.make_alt(ab, lit_c), abc);
  EXPECT_EQ(*pool.make_alt(lit_c, ab), abc);
  EXPECT_EQ(*pool.make_alt(ac, lit_b), abc);
  EXPECT_EQ(*pool.make_alt(ab, bc), abc);
  EXPECT_EQ(*pool.make_alt(abc, lit_b), abc);

  // Nullability of (L | R) when L is non-nullable and R (with higher ExprId) is
  // nullable.
  const ExprId opt_b = *pool.make_alt(lit_b, empty_str);
  const ExprId opt_bb = *pool.make_concat(opt_b, opt_b);
  EXPECT_TRUE(pool.is_nullable(*pool.make_alt(lit_a, opt_bb)));

  // Star identities: INVALID_EXPR_ID, Ø* => ε, ε* => ε, and (R*)* => R*.
  auto bad_star = pool.make_star(INVALID_EXPR_ID);
  ASSERT_FALSE(bad_star.has_value());
  EXPECT_EQ(bad_star.error(), REG_BADPAT);
  EXPECT_EQ(*pool.make_star(empty_set), empty_str);
  EXPECT_EQ(*pool.make_star(empty_str), empty_str);

  const ExprId star_a = *pool.make_star(lit_a);
  EXPECT_EQ(*pool.make_star(star_a), star_a);
  EXPECT_TRUE(pool.is_nullable(star_a));
  EXPECT_FALSE(pool.has_anchors(star_a));
  EXPECT_TRUE(pool.has_anchors(*pool.make_star(*pool.make_anchor(ANCHOR_BOL))));
  EXPECT_TRUE(Expr::make_star(lit_a) == Expr::make_star(lit_a));
  EXPECT_FALSE(Expr::make_star(lit_a) == Expr::make_star(lit_b));
}

TEST(LlvmLibcRegexExprTest, HermeticHundredPools) {
  for (size_t i = 0; i < 100; ++i) {
    ExprPool pool;
    const ExprId a = *pool.make_lit('a');
    const ExprId b = *pool.make_lit('b');
    const ExprId ab = *pool.make_concat(a, b);
    EXPECT_NE(ab, INVALID_EXPR_ID);
    EXPECT_LE(pool.allocated_bytes(), size_t(1024));
  }
}

TEST(LlvmLibcRegexExprTest, MemoryLimits) {
  ExprPool pool;
  // Verify that the pool correctly enforces MAX_NODE_LIMIT (10000 nodes).
  // EmptySet (1), EmptyStr (2), and 'a' (3) occupy the first 3 slots.
  const ExprId lit_a = *pool.make_lit('a');
  ExprId current = lit_a;
  for (size_t i = 0; i < 9997; ++i) {
    auto next = pool.make_concat(lit_a, current);
    ASSERT_TRUE(next.has_value());
    current = *next;
  }

  // Next one should fail.
  auto fail = pool.make_concat(lit_a, current);
  ASSERT_FALSE(fail.has_value());
  EXPECT_EQ(fail.error(), REG_ESPACE);

  // A new multi-bit ByteSet at MAX_NODE_LIMIT fails without incrementing
  // byte_set_count.
  ByteSet bs;
  bs.flip();
  auto bs_fail = pool.make_byte_set(bs);
  ASSERT_FALSE(bs_fail.has_value());
  EXPECT_EQ(bs_fail.error(), REG_ESPACE);
  EXPECT_EQ(pool.num_byte_sets(), size_t(0));
}
