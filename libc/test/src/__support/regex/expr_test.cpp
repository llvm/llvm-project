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
#include "src/__support/error_or.h"
#include "src/__support/regex/regex_ast.h"
#include "src/__support/regex/regex_expr_pool.h"
#include "test/UnitTest/Test.h"
#include "test/src/__support/regex/expr_builder.h"

using LIBC_NAMESPACE::regex::ANCHOR_BOL;
using LIBC_NAMESPACE::regex::ANCHOR_EOL;
using LIBC_NAMESPACE::regex::ByteSet;
using LIBC_NAMESPACE::regex::EMPTY_SET_ID;
using LIBC_NAMESPACE::regex::EMPTY_STR_ID;
using LIBC_NAMESPACE::regex::Expr;
using LIBC_NAMESPACE::regex::ExprId;
using LIBC_NAMESPACE::regex::ExprKind;
using LIBC_NAMESPACE::regex::ExprPool;
using LIBC_NAMESPACE::regex::INVALID_EXPR_ID;
using LIBC_NAMESPACE::regex::REPEAT_INF;
using LIBC_NAMESPACE::testing::ExprBuilder;

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

  auto lit_b = pool.make_lit('b');
  ASSERT_TRUE(lit_b.has_value());
  EXPECT_NE(*lit_a, *lit_b);

  auto bad_anchor = pool.make_anchor(0);
  ASSERT_FALSE(bad_anchor.has_value());
  EXPECT_EQ(bad_anchor.error(), REG_BADPAT);

  auto bol_1 = pool.make_anchor(ANCHOR_BOL);
  auto bol_2 = pool.make_anchor(ANCHOR_BOL);
  auto eol = pool.make_anchor(ANCHOR_EOL);
  ASSERT_TRUE(bol_1.has_value());
  ASSERT_TRUE(bol_2.has_value());
  ASSERT_TRUE(eol.has_value());
  EXPECT_EQ(*bol_1, *bol_2);
  EXPECT_NE(*bol_1, *eol);
  EXPECT_FALSE(pool.is_nullable(*bol_1));

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

  // Identity shortcuts on a freshly constructed pool still initialize storage
  // so the returned ExprId is safely dereferenceable via get().
  ExprPool fresh_concat;
  auto fresh_concat_id = fresh_concat.make_concat(EMPTY_STR_ID, EMPTY_STR_ID);
  ASSERT_TRUE(fresh_concat_id.has_value());
  EXPECT_EQ(*fresh_concat_id, EMPTY_STR_ID);
  EXPECT_EQ(fresh_concat.get(*fresh_concat_id).kind, ExprKind::EmptyStr);

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

  // Identity and idempotency shortcuts on a freshly constructed pool still
  // initialize storage so the returned ExprId is safely dereferenceable.
  ExprPool fresh_alt;
  auto fresh_alt_id = fresh_alt.make_alt(EMPTY_STR_ID, EMPTY_SET_ID);
  ASSERT_TRUE(fresh_alt_id.has_value());
  EXPECT_EQ(*fresh_alt_id, EMPTY_STR_ID);
  EXPECT_EQ(fresh_alt.get(*fresh_alt_id).kind, ExprKind::EmptyStr);

  ExprPool fresh_alt_lhs;
  auto fresh_alt_lhs_id = fresh_alt_lhs.make_alt(EMPTY_SET_ID, EMPTY_STR_ID);
  ASSERT_TRUE(fresh_alt_lhs_id.has_value());
  EXPECT_EQ(*fresh_alt_lhs_id, EMPTY_STR_ID);
  EXPECT_EQ(fresh_alt_lhs.get(*fresh_alt_lhs_id).kind, ExprKind::EmptyStr);

  ExprPool fresh_idem;
  auto fresh_idem_id = fresh_idem.make_alt(EMPTY_STR_ID, EMPTY_STR_ID);
  ASSERT_TRUE(fresh_idem_id.has_value());
  EXPECT_EQ(*fresh_idem_id, EMPTY_STR_ID);
  EXPECT_EQ(fresh_idem.get(*fresh_idem_id).kind, ExprKind::EmptyStr);

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
  EXPECT_TRUE(Expr::make_star(lit_a) == Expr::make_star(lit_a));
  EXPECT_FALSE(Expr::make_star(lit_a) == Expr::make_star(lit_b));

  // Repeat identities and interval coalescing in make_alt.
  const ExprId plus_a = *pool.make_repeat(lit_a, 1, REPEAT_INF);
  auto bad_rep_id = pool.make_repeat(INVALID_EXPR_ID, 1, 2);
  auto bad_rep_range = pool.make_repeat(lit_a, 3, 2);
  auto bad_rep_inf = pool.make_repeat(lit_a, REPEAT_INF, REPEAT_INF);
  ASSERT_FALSE(bad_rep_id.has_value());
  ASSERT_FALSE(bad_rep_range.has_value());
  ASSERT_FALSE(bad_rep_inf.has_value());
  EXPECT_EQ(bad_rep_id.error(), REG_BADPAT);
  EXPECT_EQ(bad_rep_range.error(), REG_BADPAT);
  EXPECT_EQ(bad_rep_inf.error(), REG_BADPAT);
  EXPECT_EQ(*pool.make_repeat(lit_a, 0, 0), empty_str);
  EXPECT_EQ(*pool.make_repeat(empty_str, 2, 4), empty_str);
  EXPECT_EQ(*pool.make_repeat(empty_set, 0, 3), empty_str);
  EXPECT_EQ(*pool.make_repeat(empty_set, 1, 3), empty_set);
  EXPECT_EQ(*pool.make_repeat(lit_a, 1, 1), lit_a);
  EXPECT_EQ(*pool.make_repeat(lit_a, 0, REPEAT_INF), star_a);
  EXPECT_EQ(*pool.make_repeat(star_a, 2, 4), star_a);
  EXPECT_EQ(*pool.make_star(plus_a), star_a);
  EXPECT_EQ(*pool.make_repeat(plus_a, 0, 3), star_a);
  EXPECT_EQ(*pool.make_repeat(plus_a, 1, 5), plus_a);
  EXPECT_EQ(*pool.make_repeat(plus_a, 1, REPEAT_INF), plus_a);
  EXPECT_EQ(*pool.make_repeat(plus_a, 2, 5),
            *pool.make_repeat(lit_a, 2, REPEAT_INF));

  const ExprId rep_0_3 = *pool.make_repeat(lit_a, 0, 3);
  const ExprId rep_1_3 = *pool.make_repeat(lit_a, 1, 3);
  const ExprId rep_2_3 = *pool.make_repeat(lit_a, 2, 3);
  const ExprId rep_3_4 = *pool.make_repeat(lit_a, 3, 4);
  const ExprId rep_4_5 = *pool.make_repeat(lit_a, 4, 5);
  const ExprId rep_5_6 = *pool.make_repeat(lit_a, 5, 6);
  const ExprId rep_1_2 = *pool.make_repeat(lit_a, 1, 2);
  const ExprId rep_1_6 = *pool.make_repeat(lit_a, 1, 6);
  const ExprId rep_2_5 = *pool.make_repeat(lit_a, 2, 5);
  EXPECT_EQ(*pool.make_alt(rep_2_3, rep_4_5), rep_2_5);
  EXPECT_EQ(*pool.make_alt(lit_a, rep_2_3), rep_1_3);
  EXPECT_EQ(*pool.make_alt(empty_str, plus_a), star_a);
  EXPECT_EQ(*pool.make_alt(*pool.make_repeat(lit_a, 2, REPEAT_INF), star_a),
            star_a);
  EXPECT_EQ(*pool.make_alt(*pool.make_concat(rep_2_3, lit_b),
                           *pool.make_concat(rep_4_5, lit_b)),
            *pool.make_concat(rep_2_5, lit_b));
  EXPECT_EQ(*pool.make_alt(*pool.make_concat(lit_b, rep_2_3),
                           *pool.make_concat(lit_b, rep_4_5)),
            *pool.make_concat(lit_b, rep_2_5));
  // Shared-suffix Concat(X, S) | S and shared-prefix Concat(P, X) | P in both
  // argument orders.
  EXPECT_EQ(*pool.make_alt(*pool.make_concat(rep_1_3, lit_b), lit_b),
            *pool.make_concat(rep_0_3, lit_b));
  EXPECT_EQ(*pool.make_alt(lit_b, *pool.make_concat(rep_1_3, lit_b)),
            *pool.make_concat(rep_0_3, lit_b));
  EXPECT_EQ(*pool.make_alt(*pool.make_concat(lit_b, rep_1_3), lit_b),
            *pool.make_concat(lit_b, rep_0_3));
  EXPECT_EQ(*pool.make_alt(lit_b, *pool.make_concat(lit_b, rep_1_3)),
            *pool.make_concat(lit_b, rep_0_3));
  // Nested Repeat(R, 2, 3) | R coalesces symmetrically when R is a Repeat node.
  const ExprId nested_rep_2_3 = *pool.make_repeat(rep_1_2, 2, 3);
  const ExprId nested_rep_1_3 = *pool.make_repeat(rep_1_2, 1, 3);
  EXPECT_EQ(*pool.make_alt(rep_1_2, nested_rep_2_3), nested_rep_1_3);
  EXPECT_EQ(*pool.make_alt(nested_rep_2_3, rep_1_2), nested_rep_1_3);
  // Out-of-order 3-way interval bridging and 2-spine merges preserve
  // commutativity and associativity.
  EXPECT_EQ(*pool.make_alt(*pool.make_alt(rep_1_2, rep_5_6), rep_3_4), rep_1_6);
  EXPECT_EQ(*pool.make_alt(rep_3_4, *pool.make_alt(rep_1_2, rep_5_6)), rep_1_6);
  const ExprId spine_empty_2_3 = *pool.make_alt(empty_str, rep_2_3);
  const ExprId spine_empty_1 = *pool.make_alt(empty_str, lit_a);
  EXPECT_EQ(*pool.make_alt(spine_empty_2_3, spine_empty_1), rep_0_3);
  EXPECT_EQ(*pool.make_alt(spine_empty_1, spine_empty_2_3), rep_0_3);
  const ExprId spine_a = *pool.make_alt(lit_c, rep_1_2);
  const ExprId spine_b = *pool.make_alt(lit_b, rep_3_4);
  EXPECT_EQ(*pool.make_alt(spine_a, spine_b), *pool.make_alt(spine_b, spine_a));
  EXPECT_EQ(*pool.make_alt(*pool.make_alt(rep_2_3, lit_c), rep_4_5),
            *pool.make_alt(rep_2_3, *pool.make_alt(lit_c, rep_4_5)));
  const ExprId merged = *pool.make_alt(*pool.make_alt(lit_c, rep_2_3), rep_1_2);
  EXPECT_EQ(merged, *pool.make_alt(lit_c, rep_1_3));
  EXPECT_LT(pool.get(merged).bin.left, pool.get(merged).bin.right);
  EXPECT_TRUE(pool.is_nullable(*pool.make_repeat(lit_a, 0, 2)));
  EXPECT_FALSE(pool.is_nullable(rep_2_3));
  EXPECT_TRUE(Expr::make_repeat(lit_a, 2, 3) == Expr::make_repeat(lit_a, 2, 3));
  EXPECT_FALSE(Expr::make_repeat(lit_a, 2, 3) ==
               Expr::make_repeat(lit_a, 2, 4));
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
  ExprId chain_24 = INVALID_EXPR_ID;
  ExprId alt_even = EMPTY_SET_ID;
  ExprId alt_odd = EMPTY_SET_ID;
  for (size_t i = 0; pool.size() < ExprPool::MAX_NODE_LIMIT; ++i) {
    auto next = pool.make_concat(lit_a, current);
    ASSERT_TRUE(next.has_value());
    current = *next;
    if (i == 23)
      chain_24 = current;
    // Build two 20-branch Alt spines (spilling ReverseExprIdStack past 16)
    // before filling the remaining pool slots.
    if (i < 40) {
      ExprId &target = (i % 2 == 0) ? alt_even : alt_odd;
      auto alt_next = pool.make_alt(target, current);
      ASSERT_TRUE(alt_next.has_value());
      target = *alt_next;
    }
  }

  // Next one should fail, including when flattening or merging chains/spines
  // that spill ReverseExprIdStack past its 16-element inline block.
  auto fail = pool.make_concat(lit_a, current);
  ASSERT_FALSE(fail.has_value());
  EXPECT_EQ(fail.error(), REG_ESPACE);
  auto spilled_fail = pool.make_concat(chain_24, current);
  ASSERT_FALSE(spilled_fail.has_value());
  EXPECT_EQ(spilled_fail.error(), REG_ESPACE);
  auto alt_insert_fail = pool.make_alt(current, alt_even);
  ASSERT_FALSE(alt_insert_fail.has_value());
  EXPECT_EQ(alt_insert_fail.error(), REG_ESPACE);
  auto alt_merge_fail = pool.make_alt(alt_even, alt_odd);
  ASSERT_FALSE(alt_merge_fail.has_value());
  EXPECT_EQ(alt_merge_fail.error(), REG_ESPACE);

  // A new multi-bit ByteSet at MAX_NODE_LIMIT fails without incrementing
  // byte_set_count.
  ByteSet bs;
  bs.flip();
  auto bs_fail = pool.make_byte_set(bs);
  ASSERT_FALSE(bs_fail.has_value());
  EXPECT_EQ(bs_fail.error(), REG_ESPACE);
  EXPECT_EQ(pool.num_byte_sets(), size_t(0));
}

TEST(LlvmLibcRegexExprTest, BoundsGuardsAndNullableMask) {
  // Uninitialised pool safely resolves constant sentinels and rejects invalid
  // or out-of-range ExprIds.
  {
    ExprPool fresh;
    EXPECT_FALSE(fresh.is_valid(EMPTY_SET_ID));
    EXPECT_TRUE(fresh.is_valid_input(EMPTY_SET_ID));
    EXPECT_TRUE(fresh.is_valid_input(EMPTY_STR_ID));
    EXPECT_FALSE(fresh.is_valid_input(INVALID_EXPR_ID));
    EXPECT_EQ(fresh.get(EMPTY_SET_ID).kind, ExprKind::EmptySet);
    EXPECT_EQ(fresh.get(EMPTY_STR_ID).kind, ExprKind::EmptyStr);
    EXPECT_FALSE(fresh.is_nullable(EMPTY_SET_ID));
    EXPECT_TRUE(fresh.is_nullable(EMPTY_STR_ID));
    EXPECT_EQ(fresh.allocated_bytes(), size_t(0));
    auto expect_badpat = [](const auto &res) {
      ASSERT_FALSE(res.has_value());
      EXPECT_EQ(res.error(), REG_BADPAT);
    };
    expect_badpat(fresh.make_star(999));
    expect_badpat(fresh.make_repeat(999, 1, 2));
    expect_badpat(fresh.make_concat(999, EMPTY_STR_ID));
    expect_badpat(fresh.make_alt(EMPTY_SET_ID, 999));
    expect_badpat(fresh.make_anchor(0x04));
    expect_badpat(fresh.make_anchor(ANCHOR_BOL | 0x04));
    expect_badpat(fresh.make_anchor(0xFF));
  }

  ExprPool pool;
  auto bol = pool.make_anchor(ANCHOR_BOL);
  auto eol = pool.make_anchor(ANCHOR_EOL);
  auto both = pool.make_anchor(ANCHOR_BOL | ANCHOR_EOL);
  auto lit_a = pool.make_lit('a');
  ASSERT_TRUE(bol.has_value());
  ASSERT_TRUE(eol.has_value());
  ASSERT_TRUE(both.has_value());
  ASSERT_TRUE(lit_a.has_value());
  auto bol_eol = pool.make_concat(*bol, *eol);
  auto bol_or_eol = pool.make_alt(*bol, *eol);
  auto bol_or_a = pool.make_alt(*bol, *lit_a);
  ASSERT_TRUE(bol_eol.has_value());
  ASSERT_TRUE(bol_or_eol.has_value());
  ASSERT_TRUE(bol_or_a.has_value());
  auto rep_bol = pool.make_repeat(*bol, 2, 3);
  auto rep_bol_0 = pool.make_repeat(*bol, 0, 3);
  auto star_bol = pool.make_star(*bol);
  auto rep_bol_or_a = pool.make_repeat(*bol_or_a, 2, 3);
  ASSERT_TRUE(rep_bol.has_value());
  ASSERT_TRUE(rep_bol_0.has_value());
  ASSERT_TRUE(star_bol.has_value());
  ASSERT_TRUE(rep_bol_or_a.has_value());
  EXPECT_EQ(pool.get(*bol).nullable_mask, uint8_t(0b1010));
  EXPECT_EQ(pool.get(*eol).nullable_mask, uint8_t(0b1100));
  EXPECT_EQ(pool.get(*both).nullable_mask, uint8_t(0b1000));
  EXPECT_EQ(pool.get(*bol_eol).nullable_mask, uint8_t(0b1000));
  EXPECT_EQ(pool.get(*bol_or_eol).nullable_mask, uint8_t(0b1110));
  EXPECT_EQ(pool.get(*rep_bol).nullable_mask, uint8_t(0b1010));
  EXPECT_EQ(pool.get(*rep_bol_0).nullable_mask, uint8_t(0b1111));
  EXPECT_EQ(pool.get(*star_bol).nullable_mask, uint8_t(0b1111));
  EXPECT_EQ(pool.get(*rep_bol_or_a).nullable_mask, uint8_t(0b1010));
}

TEST(LlvmLibcRegexExprTest, DeepConcatFlatteningAndPrefixCoalesce) {
  ExprBuilder b;
  const ExprId lit_d = b.lit('d');

  // Flatten a 500-deep asymmetric left Concat spine (spilling
  // ReverseExprIdStack's 16-item inline block): P · d == a · (b · (c · ... d)).
  constexpr size_t DEPTH = 500;
  const ExprId prefix = b.cycle_literals("abc", DEPTH);
  const ExprId expected = b.cycle_literals("abc", DEPTH, lit_d);
  EXPECT_EQ(b.cat(prefix, lit_d), expected);

  // Coalesce 500-deep shared-prefix Concat spines in both argument orders:
  //   (P · d{1,2}) | (P · d{3,4}) -> P · d{1,4}
  //   (P · d{1,2}) | (P · d{4,5}) -> Alt (non-adjacent, unmerged)
  const ExprId p_1_2 = b.cycle_literals("abc", DEPTH, b.rep(lit_d, 1, 2));
  const ExprId p_3_4 = b.cycle_literals("abc", DEPTH, b.rep(lit_d, 3, 4));
  const ExprId p_4_5 = b.cycle_literals("abc", DEPTH, b.rep(lit_d, 4, 5));
  const ExprId p_1_4 = b.cycle_literals("abc", DEPTH, b.rep(lit_d, 1, 4));
  EXPECT_EQ(b.alt(p_1_2, p_3_4), p_1_4);
  EXPECT_EQ(b.alt(p_3_4, p_1_2), p_1_4);
  EXPECT_EQ(b.raw().get(b.alt(p_1_2, p_4_5)).kind, ExprKind::Alt);
}

TEST(LlvmLibcRegexExprTest, NestedRepeatStarUnwrapping) {
  ExprBuilder b;
  const ExprId lit_a = b.lit('a');

  // Unwrap nested repeats with min <= 1: ((a{1,2}){1,3})* -> a*
  const ExprId rep_1_2 = b.rep(lit_a, 1, 2);
  EXPECT_EQ(b.star(b.rep(rep_1_2, 1, 3)), b.star(lit_a));

  // Stop unwrapping once min >= 2: ((a{2,3}){0,2})* -> (a{2,3})*
  const ExprId rep_2_3 = b.rep(lit_a, 2, 3);
  EXPECT_EQ(b.star(b.rep(rep_2_3, 0, 2)), b.star(rep_2_3));
}

TEST(LlvmLibcRegexExprTest, WideAltSpineInsertAndBridging) {
  ExprBuilder b;
  constexpr size_t BRANCH_COUNT = 600;
  constexpr size_t ASC_COUNT = 64;
  // Pre-intern 600 distinct branches in ascending ExprId order.
  for (size_t i = 0; i < BRANCH_COUNT; ++i)
    ASSERT_NE(b.distinct_branch(i), INVALID_EXPR_ID) << "iteration " << i;

  const ExprId z_1_2 = b.rep('z', 1, 2);
  const ExprId z_3_4 = b.rep('z', 3, 4);
  const ExprId z_5_6 = b.rep('z', 5, 6);
  const ExprId z_1_6 = b.rep('z', 1, 6);

  // Ascending insertion onto a repeat-bearing spine exercises the formerly
  // cubic path (scanning past growing prefixes and spilling prefix in
  // insert_sorted_no_coalesce) and matches descending construction.
  ExprId asc_spine = z_1_2;
  ExprId desc_64 = z_1_2;
  for (size_t i = 0; i < ASC_COUNT; ++i) {
    asc_spine = b.alt(b.distinct_branch(i), asc_spine);
    desc_64 = b.alt(b.distinct_branch(ASC_COUNT - 1 - i), desc_64);
    ASSERT_NE(asc_spine, INVALID_EXPR_ID) << "iteration " << i;
  }
  EXPECT_EQ(asc_spine, desc_64);
  EXPECT_EQ(b.verify_alt_spine(asc_spine), ASC_COUNT + 1);

  // Build a 601-branch spine in descending order (O(BRANCH_COUNT) pool nodes)
  // so bridging and tail insertion spill ReverseExprIdStack across 38 blocks.
  ExprId spine = z_1_2;
  for (size_t i = BRANCH_COUNT; i > 0; --i) {
    spine = b.alt(b.distinct_branch(i - 1), spine);
    ASSERT_NE(spine, INVALID_EXPR_ID) << "iteration " << i;
  }

  // Bridging z{1,2} | z{5,6} | z{3,4} -> z{1,6} across 601 branches commutes,
  // stays strictly sorted, and replaces all three repeats.
  const ExprId with_5_6 = b.alt(spine, z_5_6);
  const ExprId bridged_lr = b.alt(with_5_6, z_3_4);
  const ExprId bridged_rl = b.alt(z_3_4, with_5_6);
  EXPECT_EQ(bridged_lr, bridged_rl);
  EXPECT_EQ(b.verify_alt_spine(bridged_lr), BRANCH_COUNT + 1);
  EXPECT_EQ(b.count_branch(bridged_lr, z_1_6), size_t(1));
  EXPECT_EQ(b.count_branch(bridged_lr, z_1_2), size_t(0));
  EXPECT_EQ(b.count_branch(bridged_lr, z_3_4), size_t(0));
  EXPECT_EQ(b.count_branch(bridged_lr, z_5_6), size_t(0));

  // Appending a fresh higher-ID branch spills all 601 prefix entries.
  const ExprId tail_appended = b.alt(bridged_lr, b.lit('~'));
  EXPECT_EQ(b.verify_alt_spine(tail_appended), BRANCH_COUNT + 2);
}

TEST(LlvmLibcRegexExprTest, TwoSpineAltMergeSpillAndCommutativity) {
  ExprBuilder b;

  // Merge two 48-branch Alt spines (spilling ReverseExprIdStack's 16-item
  // inline block) with shared branches, both plain and with bridged Repeat
  // intervals: (L | z{1,2} | z{5,6}) | (R | z{3,4}) -> L | R | z{1,6}.
  ExprId left_plain = EMPTY_SET_ID;
  ExprId right_plain = EMPTY_SET_ID;
  for (size_t i = 0; i < 24; ++i) {
    const ExprId shared = b.distinct_branch(60 + i);
    left_plain = b.alt(left_plain, b.distinct_branch(i * 2), shared);
    right_plain = b.alt(right_plain, b.distinct_branch(i * 2 + 1), shared);
    ASSERT_NE(left_plain, INVALID_EXPR_ID) << "iteration " << i;
    ASSERT_NE(right_plain, INVALID_EXPR_ID) << "iteration " << i;
  }
  const ExprId plain_lr = b.alt(left_plain, right_plain);
  EXPECT_EQ(plain_lr, b.alt(right_plain, left_plain));
  EXPECT_EQ(b.alt(left_plain, left_plain), left_plain);

  const ExprId left_rep = b.alt(left_plain, b.rep('z', 1, 2), b.rep('z', 5, 6));
  const ExprId right_rep = b.alt(right_plain, b.rep('z', 3, 4));
  const ExprId rep_want = b.alt(plain_lr, b.rep('z', 1, 6));
  EXPECT_EQ(b.alt(left_rep, right_rep), rep_want);
  EXPECT_EQ(b.alt(right_rep, left_rep), rep_want);

  // Cross-spine Repeat merges that break commutativity if one spine is
  // flattened asymmetrically into the other:
  //   (a | a*) | (ε | a{2,5})      -> ε | a*
  //   (ε | a)  | (b{1,2} | a{1,2}) commutes in both orders
  const ExprId a_or_star = b.alt(b.lit('a'), b.star('a'));
  const ExprId eps_or_a25 = b.alt(EMPTY_STR_ID, b.rep('a', 2, 5));
  const ExprId eps_or_star = b.alt(EMPTY_STR_ID, b.star('a'));
  EXPECT_EQ(b.alt(a_or_star, eps_or_a25), eps_or_star);
  EXPECT_EQ(b.alt(eps_or_a25, a_or_star), eps_or_star);

  const ExprId eps_or_a = b.alt(EMPTY_STR_ID, b.lit('a'));
  const ExprId b12_or_a12 = b.alt(b.rep('b', 1, 2), b.rep('a', 1, 2));
  EXPECT_EQ(b.alt(eps_or_a, b12_or_a12), b.alt(b12_or_a12, eps_or_a));

  // Disjoint two-spine merge where low_alt's ExprId precedes high_alt's head.
  const ExprId low_alt = b.alt(b.distinct_branch(100), b.distinct_branch(101));
  const ExprId high_alt = b.alt(b.distinct_branch(102), b.distinct_branch(103));
  ASSERT_LT(low_alt, b.get(high_alt).bin.left);
  const ExprId merged_disjoint = b.alt(low_alt, high_alt);
  EXPECT_EQ(merged_disjoint, b.alt(high_alt, low_alt));
  EXPECT_EQ(b.verify_alt_spine(merged_disjoint), size_t(4));
}

TEST(LlvmLibcRegexExprTest, AltCoalesceCascadesBetweenKeptAndTail) {
  ExprBuilder b;

  // Absorbing earlier ε widens a{1,3} -> a{0,3}, unlocking later b{1,2}·a{0,3}:
  //   a{2,3} | (ε | a | b{1,2}·a{0,3}) -> b{0,2}·a{0,3}.
  const ExprId concat_spine = b.alt(EMPTY_STR_ID, b.lit('a'),
                                    b.cat(b.rep('b', 1, 2), b.rep('a', 0, 3)));
  const ExprId concat_want = b.cat(b.rep('b', 0, 2), b.rep('a', 0, 3));
  EXPECT_EQ(b.alt(b.rep('a', 2, 3), concat_spine), concat_want);
  EXPECT_EQ(b.alt(concat_spine, b.rep('a', 2, 3)), concat_want);

  // Absorbing earlier ε turns a{1,∞} into a*, unlocking later b{1,2}·a*:
  //   a{2,2} | (ε | a | a{3,∞} | b{1,2}·a*) -> b{0,2}·a*.
  const ExprId star_spine =
      b.alt(EMPTY_STR_ID, b.lit('a'), b.rep('a', 3, REPEAT_INF),
            b.cat(b.rep('b', 1, 2), b.star('a')));
  EXPECT_EQ(b.alt(b.rep('a', 2, 2), star_spine),
            b.cat(b.rep('b', 0, 2), b.star('a')));

  // Absorbing earlier ε widens a{1,3} -> a{0,3}, unlocking nested
  // (a{0,3}){2,3}:
  //   a{2,3} | (ε | a | (a{0,3}){2,3}) -> (a{0,3}){1,3}.
  const ExprId a_0_3 = b.rep('a', 0, 3);
  const ExprId nested_spine =
      b.alt(EMPTY_STR_ID, b.lit('a'), b.rep(a_0_3, 2, 3));
  EXPECT_EQ(b.alt(b.rep('a', 2, 3), nested_spine), b.rep(a_0_3, 1, 3));

  // Preserving an unrelated middle branch m while bridging z{1,2} and z{5,6}:
  //   (z{1,2} | m | z{5,6}) | z{3,4} -> m | z{1,6}.
  const ExprId z_1_2 = b.rep('z', 1, 2);
  const ExprId m = b.lit('m');
  const ExprId z_5_6 = b.rep('z', 5, 6);
  ASSERT_LT(z_1_2, m);
  ASSERT_LT(m, z_5_6);
  EXPECT_EQ(b.alt(b.alt(z_1_2, m, z_5_6), b.rep('z', 3, 4)),
            b.alt(m, b.rep('z', 1, 6)));
}

TEST(LlvmLibcRegexExprTest, AltCoalescedBranchPlacement) {
  ExprBuilder b;
  // Intern in ascending ExprId order (a < b* < d < b{2,3} < f < b{1,3}) so
  // coalesced results land at each sorted position along the spine.
  const ExprId a = b.lit('a');
  const ExprId star_b = b.star('b');
  const ExprId d = b.lit('d');
  const ExprId b_2_3 = b.rep('b', 2, 3);
  const ExprId f = b.lit('f');
  const ExprId b_1_3 = b.rep('b', 1, 3);
  ASSERT_LT(a, star_b);
  ASSERT_LT(star_b, d);
  ASSERT_LT(d, b_2_3);
  ASSERT_LT(b_2_3, f);
  ASSERT_LT(f, b_1_3);

  // Coalesced branch lands at the front: (d | b{2,3}) | b* -> b* | d.
  EXPECT_EQ(b.alt(b.alt(d, b_2_3), star_b), b.alt(star_b, d));

  // Coalesced branch lands between earlier skipped branches:
  //   (a | d | b{2,3}) | b*     -> a | b* | d
  //   (a | d | b{2,3} | f) | b* -> a | b* | d | f
  EXPECT_EQ(b.alt(b.alt(a, d, b_2_3), star_b), b.alt(a, star_b, d));
  EXPECT_EQ(b.alt(b.alt(a, d, b_2_3, f), star_b), b.alt(a, star_b, d, f));

  // Coalesced branch sorts past a later branch:
  //   (a | b{2,3} | f) | b -> a | f | b{1,3}.
  EXPECT_EQ(b.alt(b.alt(a, b_2_3, f), b.lit('b')), b.alt(a, f, b_1_3));

  // Coalesced branch lands right between skipped and remaining branches
  // (k < c{0,5} < c{0,2} < w < c{3,5}).
  const ExprId k = b.lit('k');
  const ExprId c_0_5 = b.rep('c', 0, 5);
  const ExprId c_0_2 = b.rep('c', 0, 2);
  const ExprId w = b.lit('w');
  const ExprId c_3_5 = b.rep('c', 3, 5);
  ASSERT_LT(k, c_0_5);
  ASSERT_LT(c_0_5, w);
  EXPECT_EQ(b.alt(c_3_5, b.alt(c_0_2, w)), b.alt(c_0_5, w));
  EXPECT_EQ(b.alt(c_3_5, b.alt(k, c_0_2, w)), b.alt(k, c_0_5, w));

  // Coalescing unwraps to a branch already later in the spine (q* and
  // (q{2,3})*): (R{0,3}){0,1} | ((R{0,3}){2,∞} | R*) -> R*.
  const ExprId q_0_3 = b.rep('q', 0, 3);
  const ExprId q03_2_inf = b.rep(q_0_3, 2, REPEAT_INF);
  const ExprId star_q = b.star('q');
  ASSERT_LT(q03_2_inf, star_q);
  EXPECT_EQ(b.alt(b.rep(q_0_3, 0, 1), b.alt(q03_2_inf, star_q)), star_q);

  const ExprId q_2_3 = b.rep('q', 2, 3);
  const ExprId r_0_3 = b.rep(q_2_3, 0, 3);
  const ExprId r03_2_inf = b.rep(r_0_3, 2, REPEAT_INF);
  const ExprId star_q23 = b.star(q_2_3);
  ASSERT_LT(r03_2_inf, star_q23);
  EXPECT_EQ(b.alt(b.rep(r_0_3, 0, 1), b.alt(r03_2_inf, star_q23)), star_q23);

  // Coalescing unwraps to a branch already earlier in the spine
  // (p* < (p{0,3}){2,∞}): (p{0,3}){0,1} | (p* | (p{0,3}){2,∞}) -> p*.
  const ExprId star_p = b.star('p');
  const ExprId p_0_3 = b.rep('p', 0, 3);
  const ExprId p03_2_inf = b.rep(p_0_3, 2, REPEAT_INF);
  ASSERT_LT(star_p, p03_2_inf);
  EXPECT_EQ(b.alt(b.rep(p_0_3, 0, 1), b.alt(star_p, p03_2_inf)), star_p);
}
