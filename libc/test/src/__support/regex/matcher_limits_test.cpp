//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Deep-chain tests for iterative derivative evaluation.
///
/// These tests build expression chains thousands of nodes deep.  They live in
/// their own binary because the hermetic test heap is a fixed 1 MiB that is
/// never freed, and matcher_test already uses most of it.
///
//===----------------------------------------------------------------------===//

#include "hdr/regex_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/regex/regex_ast.h"
#include "src/__support/regex/regex_matcher.h"
#include "src/__support/regex/regex_parser.h"
#include "test/UnitTest/Test.h"
#include "test/src/__support/regex/expr_builder.h"

using LIBC_NAMESPACE::regex::derivative;
using LIBC_NAMESPACE::regex::EMPTY_SET_ID;
using LIBC_NAMESPACE::regex::EMPTY_STR_ID;
using LIBC_NAMESPACE::regex::ExprId;
using LIBC_NAMESPACE::regex::match;
using LIBC_NAMESPACE::regex::MatchResult;
using LIBC_NAMESPACE::regex::parse_ere;
using LIBC_NAMESPACE::regex::REPEAT_INF;
using LIBC_NAMESPACE::testing::ExprBuilder;

namespace {

// Builds the right-associated chain (a?)(a?)...(a?)b with `count` optional a's.
ExprId build_nullable_a_chain(ExprBuilder &b, size_t count) {
  const ExprId opt_a = b.alt(EMPTY_STR_ID, b.lit('a'));
  ExprId chain = b.lit('b');
  for (size_t i = 0; i < count; ++i)
    chain = b.cat(opt_a, chain);
  return chain;
}

ExprId deriv(ExprBuilder &b, ExprId id, char ch) {
  return ExprBuilder::unwrap(derivative(id, ch, b.raw()));
}

} // namespace

TEST(LlvmLibcRegexMatcherLimitsTest, DeepNullableConcatChain) {
  // A chain of 2,000 optional 'a's followed by 'b'.  Walking it needs about
  // 2,000 stack frames, far more than DerivStack keeps inline, so the stack
  // moves to the heap and grows several times.
  ExprBuilder b;
  const ExprId chain = build_nullable_a_chain(b, 2000);
  EXPECT_EQ(deriv(b, chain, 'b'), EMPTY_STR_ID);
  EXPECT_EQ(deriv(b, chain, 'x'), EMPTY_SET_ID);
  EXPECT_EQ(match(b.raw(), chain, "b"), MatchResult::Match);
  EXPECT_EQ(match(b.raw(), chain, "xxxx"), MatchResult::NoMatch);
}

TEST(LlvmLibcRegexMatcherLimitsTest, MultiByteChainReuseAndAltDerivative) {
  // A 24-element chain matched over several bytes.  The stack outgrows its
  // inline buffer on the first byte and match() keeps using it for the rest.
  // D_a produces a non-trivial 24-branch alternation across the grown stack.
  ExprBuilder b;
  const ExprId chain = build_nullable_a_chain(b, 24);
  EXPECT_EQ(b.verify_alt_spine(deriv(b, chain, 'a')), size_t(24));
  EXPECT_EQ(match(b.raw(), chain, "b"), MatchResult::Match);
  EXPECT_EQ(match(b.raw(), chain, "aaab"), MatchResult::Match);
  EXPECT_EQ(match(b.raw(), chain, "xxxx"), MatchResult::NoMatch);
}

TEST(LlvmLibcRegexMatcherLimitsTest, DeepAltSpine) {
  // 64 two-letter branches: 0..59 start with 'a', 60..63 with 'b'.  The walk
  // goes down the right side of every Alt with that Alt's left result saved.
  ExprBuilder b;
  ExprId spine = EMPTY_SET_ID;
  for (size_t i = 0; i < 64; ++i)
    spine = b.alt(spine, b.distinct_branch(i));
  EXPECT_EQ(b.verify_alt_spine(deriv(b, spine, 'a')), size_t(60));
  EXPECT_EQ(b.verify_alt_spine(deriv(b, spine, 'b')), size_t(4));
  EXPECT_EQ(deriv(b, spine, 'x'), EMPTY_SET_ID);
}

TEST(LlvmLibcRegexMatcherLimitsTest, AsymmetricNestedStackOrder) {
  // 10 nested (expr{1,2} . ch) wrappers around 'a' (depth 20 down the left
  // spine, growing DerivStack twice) with distinct suffixes 'b'..'k'.
  // Reversing or scrambling copied frames on stack growth produces the wrong
  // suffix order and fails the match checks.
  ExprBuilder b;
  ExprId expr = b.lit('a');
  for (size_t i = 0; i < 10; ++i)
    expr = b.cat(b.rep(expr, 1, 2), b.lit(static_cast<char>('b' + i)));
  EXPECT_EQ(match(b.raw(), expr, "abcdefghijk"), MatchResult::Match);
  EXPECT_EQ(match(b.raw(), expr, "akjihgfedcb"), MatchResult::NoMatch);
  EXPECT_EQ(match(b.raw(), expr, "abcdefghij"), MatchResult::NoMatch);
}

TEST(LlvmLibcRegexMatcherLimitsTest, RepeatDerivativeBoundaryRules) {
  ExprBuilder b;

  // Unbounded repeat preserves REPEAT_INF across differentiation:
  // D_a(a{3,inf}) == a{2,inf}.
  EXPECT_EQ(deriv(b, b.rep('a', 3, REPEAT_INF), 'a'),
            b.rep('a', 2, REPEAT_INF));

  // Conditionally nullable repeat body with a non-nullable suffix: at byte 0
  // ('^' matches), (^|a) is nullable so leading copies can match empty and
  // D_a((^|a){2,3}) drops min to 0, letting "ab" match.  Without '^'
  // (REG_NOTBOL or after a leading 'x'), (^|a) is not nullable so min only
  // drops to 1, rejecting "ab"/"xab" and requiring "aab".
  const ExprId rep_b =
      ExprBuilder::unwrap(parse_ere("(^|a){2,3}b", REG_EXTENDED, b.raw()));
  EXPECT_EQ(match(b.raw(), rep_b, "ab"), MatchResult::Match);
  EXPECT_EQ(match(b.raw(), rep_b, "ab", REG_EXTENDED, REG_NOTBOL),
            MatchResult::NoMatch);
  EXPECT_EQ(match(b.raw(), rep_b, "xab"), MatchResult::NoMatch);
  EXPECT_EQ(match(b.raw(), rep_b, "aab", REG_EXTENDED, REG_NOTBOL),
            MatchResult::Match);
}
