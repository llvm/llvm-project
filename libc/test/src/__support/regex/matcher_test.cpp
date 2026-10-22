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
#include "src/__support/regex/regex_parser.h"
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
using LIBC_NAMESPACE::regex::parse_ere;

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
  ErrorOr<ExprId> hello = parse_ere("hello", REG_EXTENDED, pool);
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
  ErrorOr<ExprId> pat = parse_ere("ababac", REG_EXTENDED, pool);
  ASSERT_TRUE(pat.has_value());

  // At index 4 ('a'), the partial match "ababa" fails on 'b' at index 5 while
  // a second partial match started at index 2 ("aba") advances to "ababac".
  EXPECT_EQ(match(pool, *pat, "abababac"), MatchResult::Match);
  EXPECT_EQ(match(pool, *pat, "abababab"), MatchResult::NoMatch);
}

TEST(LlvmLibcRegexMatcherTest, WildcardMatchingAndNewlineFlag) {
  ExprPool pool;
  ErrorOr<ExprId> pat = parse_ere("a.c", REG_EXTENDED, pool);
  ASSERT_TRUE(pat.has_value());
  EXPECT_EQ(pool.num_byte_sets(), size_t(1));
  EXPECT_LE(pool.allocated_bytes(), size_t(1024));

  EXPECT_EQ(match(pool, *pat, "xx abc yy"), MatchResult::Match);
  EXPECT_EQ(match(pool, *pat, "a\nc"), MatchResult::Match);
  EXPECT_EQ(match(pool, *pat,
                  "a\xff"
                  "c"),
            MatchResult::Match);
  EXPECT_EQ(match(pool, *pat, "ac"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *pat, "abbc"), MatchResult::NoMatch);

  const size_t warm_nodes = pool.size();
  const size_t warm_bytes = pool.allocated_bytes();
  EXPECT_EQ(match(pool, *pat, "xx abc yy"), MatchResult::Match);
  EXPECT_EQ(pool.size(), warm_nodes);
  EXPECT_EQ(pool.allocated_bytes(), warm_bytes);

  // A second '.' wildcard in the same mode shares the existing ByteSet slot.
  ErrorOr<ExprId> pat2 = parse_ere("a.b.c", REG_EXTENDED, pool);
  ASSERT_TRUE(pat2.has_value());
  EXPECT_EQ(pool.num_byte_sets(), size_t(1));
  EXPECT_EQ(match(pool, *pat2, "axbyc"), MatchResult::Match);

  // Under REG_NEWLINE, '.' matches any byte except '\n'.
  ErrorOr<ExprId> nl_pat = parse_ere("a.c", REG_EXTENDED | REG_NEWLINE, pool);
  ASSERT_TRUE(nl_pat.has_value());
  EXPECT_EQ(pool.num_byte_sets(), size_t(2));
  EXPECT_EQ(match(pool, *nl_pat, "abc"), MatchResult::Match);
  EXPECT_EQ(match(pool, *nl_pat, "a\nc"), MatchResult::NoMatch);
}

TEST(LlvmLibcRegexMatcherTest, AnchorMatchingAndFlags) {
  ExprPool pool;
  ErrorOr<ExprId> pat = parse_ere("^foo$", REG_EXTENDED, pool);
  ASSERT_TRUE(pat.has_value());
  EXPECT_LE(pool.size(), size_t(12));
  EXPECT_LE(pool.allocated_bytes(), size_t(1024));

  EXPECT_EQ(match(pool, *pat, "foo"), MatchResult::Match);
  const size_t warm_nodes = pool.size();
  const size_t warm_bytes = pool.allocated_bytes();
  EXPECT_EQ(match(pool, *pat, "foo"), MatchResult::Match);
  EXPECT_EQ(pool.size(), warm_nodes);
  EXPECT_EQ(pool.allocated_bytes(), warm_bytes);

  // Near-miss negative checks for ^foo$.
  EXPECT_EQ(match(pool, *pat, "xfoo"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *pat, "foox"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *pat, "foo\nbar"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *pat, ""), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *pat, "foo", 0, REG_NOTBOL), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *pat, "foo", 0, REG_NOTEOL), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *pat, "bar\nfoo\nbaz", REG_NEWLINE, 0),
            MatchResult::Match);

  // Empty-line and multi-anchor patterns.
  ErrorOr<ExprId> empty_line = parse_ere("^$", REG_EXTENDED, pool);
  ErrorOr<ExprId> multi_bol = parse_ere("^^ab$$", REG_EXTENDED, pool);
  ErrorOr<ExprId> mid_bol = parse_ere("a^b", REG_EXTENDED, pool);
  ErrorOr<ExprId> nl_bol = parse_ere("\n^b", REG_EXTENDED, pool);
  ASSERT_TRUE(empty_line.has_value());
  ASSERT_TRUE(multi_bol.has_value());
  ASSERT_TRUE(mid_bol.has_value());
  ASSERT_TRUE(nl_bol.has_value());

  EXPECT_EQ(match(pool, *empty_line, ""), MatchResult::Match);
  EXPECT_EQ(match(pool, *empty_line, "a"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *empty_line, "\n"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *empty_line, "a\n\nb", REG_NEWLINE, 0),
            MatchResult::Match);
  EXPECT_EQ(match(pool, *empty_line, "a\nb", REG_NEWLINE, 0),
            MatchResult::NoMatch);

  EXPECT_EQ(match(pool, *multi_bol, "ab"), MatchResult::Match);
  EXPECT_EQ(match(pool, *multi_bol, "xab"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *multi_bol, "abx"), MatchResult::NoMatch);

  EXPECT_EQ(match(pool, *mid_bol, "ab"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *mid_bol, "a^b"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *nl_bol, "\nb"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *nl_bol, "\nb", REG_NEWLINE, 0), MatchResult::Match);
}

TEST(LlvmLibcRegexMatcherTest, ParseEreEscapesAndErrors) {
  ExprPool pool;
  ErrorOr<ExprId> escaped_dot = parse_ere(R"(a\.b)", REG_EXTENDED, pool);
  ErrorOr<ExprId> escaped_star = parse_ere(R"(a\*b)", REG_EXTENDED, pool);
  ErrorOr<ExprId> escaped_slash = parse_ere(R"(a\\b)", REG_EXTENDED, pool);
  ErrorOr<ExprId> escaped_anchors = parse_ere(R"(a\^b\$c)", REG_EXTENDED, pool);
  ASSERT_TRUE(escaped_dot.has_value());
  ASSERT_TRUE(escaped_star.has_value());
  ASSERT_TRUE(escaped_slash.has_value());
  ASSERT_TRUE(escaped_anchors.has_value());

  EXPECT_EQ(match(pool, *escaped_dot, "a.b"), MatchResult::Match);
  EXPECT_EQ(match(pool, *escaped_dot, "axb"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *escaped_star, "a*b"), MatchResult::Match);
  EXPECT_EQ(match(pool, *escaped_slash, R"(a\b)"), MatchResult::Match);
  EXPECT_EQ(match(pool, *escaped_anchors, "a^b$c"), MatchResult::Match);
  EXPECT_EQ(match(pool, *escaped_anchors, "axb$c"), MatchResult::NoMatch);

  ErrorOr<ExprId> trailing_escape = parse_ere(R"(abc\)", REG_EXTENDED, pool);
  ASSERT_FALSE(trailing_escape.has_value());
  EXPECT_EQ(trailing_escape.error(), REG_EESCAPE);

  const string_view bad_patterns[] = {
      "*",     "+",     "?",     "(",     ")",     "[",     "{",     "|",
      R"(\1)", R"(\2)", R"(\3)", R"(\4)", R"(\5)", R"(\6)", R"(\7)", R"(\8)",
      R"(\9)", R"(\b)", R"(\B)", R"(\<)", R"(\>)", R"(\A)", R"(\Z)", R"(\`)",
      R"(\')", R"(\w)", R"(\W)", R"(\s)", R"(\S)", R"(\d)", R"(\D)"};
  for (string_view bad : bad_patterns) {
    ErrorOr<ExprId> err = parse_ere(bad, REG_EXTENDED, pool);
    ASSERT_FALSE(err.has_value());
    EXPECT_EQ(err.error(), REG_BADPAT);
  }
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

  ErrorOr<ExprId> dot_exhausted = parse_ere(".", REG_EXTENDED, pool);
  ASSERT_FALSE(dot_exhausted.has_value());
  EXPECT_EQ(dot_exhausted.error(), REG_ESPACE);

  ErrorOr<ExprId> concat_exhausted = parse_ere("ba", REG_EXTENDED, pool);
  ASSERT_FALSE(concat_exhausted.has_value());
  EXPECT_EQ(concat_exhausted.error(), REG_ESPACE);
}
