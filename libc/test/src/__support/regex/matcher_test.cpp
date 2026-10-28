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

TEST(LlvmLibcRegexMatcherTest, StarMatchingAndStabilization) {
  ExprPool pool;
  ErrorOr<ExprId> pat = parse_ere("ab*c", REG_EXTENDED, pool);
  ASSERT_TRUE(pat.has_value());
  EXPECT_LE(pool.allocated_bytes(), size_t(1024));

  EXPECT_EQ(match(pool, *pat, "ac"), MatchResult::Match);
  EXPECT_EQ(match(pool, *pat, "abc"), MatchResult::Match);
  EXPECT_EQ(match(pool, *pat, "abbbc"), MatchResult::Match);
  // Near-miss negative checks for ab*c.
  EXPECT_EQ(match(pool, *pat, "abbd"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *pat, "bbbc"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *pat, "abXc"), MatchResult::NoMatch);

  // State-space stabilization canary: scanning "a.*c" across a 10,000-byte
  // input must converge to a small fixed set of canonical derivative states
  // within the initial 1 KiB pool.
  ExprPool star_pool;
  ErrorOr<ExprId> dot_star = parse_ere("a.*c", REG_EXTENDED, star_pool);
  ASSERT_TRUE(dot_star.has_value());
  char long_input[10000];
  long_input[0] = 'a';
  for (size_t i = 1; i < 9999; ++i)
    long_input[i] = 'b';
  long_input[9999] = 'c';
  EXPECT_EQ(match(star_pool, *dot_star, string_view(long_input, 10000)),
            MatchResult::Match);
  EXPECT_LE(star_pool.size(), size_t(16));
  EXPECT_LE(star_pool.allocated_bytes(), size_t(1024));

  ErrorOr<ExprId> nl_dot_star =
      parse_ere("a.*c", REG_EXTENDED | REG_NEWLINE, star_pool);
  ASSERT_TRUE(nl_dot_star.has_value());
  EXPECT_EQ(match(star_pool, *nl_dot_star, "abbc", REG_NEWLINE, 0),
            MatchResult::Match);
  EXPECT_EQ(match(star_pool, *nl_dot_star, "a\nc", REG_NEWLINE, 0),
            MatchResult::NoMatch);
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

  const string_view bad_rpt_patterns[] = {"*",   "+",   "?",    "{1}", "^*",
                                          "^+",  "^?",  "^{1}", "$*",  "a**",
                                          "a*+", "a*?", "a*{1}"};
  for (string_view bad_rpt : bad_rpt_patterns) {
    ErrorOr<ExprId> err = parse_ere(bad_rpt, REG_EXTENDED, pool);
    ASSERT_FALSE(err.has_value());
    EXPECT_EQ(err.error(), REG_BADRPT);
  }

  const string_view ebrack_patterns[] = {
      "[", "[^", "[abc", "[]", "a[[.x", "a[[.x.", "a[[:digit:", "a[[:digit"};
  for (string_view pat : ebrack_patterns) {
    ErrorOr<ExprId> err = parse_ere(pat, REG_EXTENDED, pool);
    ASSERT_FALSE(err.has_value());
    EXPECT_EQ(err.error(), REG_EBRACK);
  }

  const string_view erange_patterns[] = {"a[3-1]c", "a[1-3-5]c", "a[1-",
                                         "a[a-[:digit:]]c", "a[[:alpha:]-z]c"};
  for (string_view pat : erange_patterns) {
    ErrorOr<ExprId> err = parse_ere(pat, REG_EXTENDED, pool);
    ASSERT_FALSE(err.has_value());
    EXPECT_EQ(err.error(), REG_ERANGE);
  }

  const string_view ecollate_patterns[] = {"a[[.notdef.]]b", "a[[.x,.]]",
                                           "a[[=b,=]]"};
  for (string_view pat : ecollate_patterns) {
    ErrorOr<ExprId> err = parse_ere(pat, REG_EXTENDED, pool);
    ASSERT_FALSE(err.has_value());
    EXPECT_EQ(err.error(), REG_ECOLLATE);
  }

  const string_view ectype_patterns[] = {"a[[:notdef:]]b", "a[[:alph:]]b",
                                         "a[[:]:]]b", "a[[:-:]]b"};
  for (string_view pat : ectype_patterns) {
    ErrorOr<ExprId> err = parse_ere(pat, REG_EXTENDED, pool);
    ASSERT_FALSE(err.has_value());
    EXPECT_EQ(err.error(), REG_ECTYPE);
  }

  const string_view bad_patterns[] = {
      // Unimplemented quantifiers.
      "a+", "a?", "a{1}", "{",
      // Unimplemented word boundaries.
      "[[:<:]]", "[[:>:]]",
      // Unimplemented backreferences.
      R"(\1)", R"(\2)", R"(\3)", R"(\4)", R"(\5)", R"(\6)", R"(\7)", R"(\8)",
      R"(\9)",
      // Unimplemented boundary assertions.
      R"(\b)", R"(\B)", R"(\<)", R"(\>)", R"(\A)", R"(\Z)", R"(\`)", R"(\')"};
  for (string_view bad : bad_patterns) {
    ErrorOr<ExprId> err = parse_ere(bad, REG_EXTENDED, pool);
    ASSERT_FALSE(err.has_value());
    EXPECT_EQ(err.error(), REG_BADPAT);
  }

  ErrorOr<ExprId> empty_pipe = parse_ere("|", REG_EXTENDED, pool);
  ASSERT_FALSE(empty_pipe.has_value());
  EXPECT_EQ(empty_pipe.error(), REG_BADPAT);

  ErrorOr<ExprId> unclosed_paren = parse_ere("(", REG_EXTENDED, pool);
  ASSERT_FALSE(unclosed_paren.has_value());
  EXPECT_EQ(unclosed_paren.error(), REG_EPAREN);

  ErrorOr<ExprId> literal_close = parse_ere(")", REG_EXTENDED, pool);
  ASSERT_TRUE(literal_close.has_value());
  EXPECT_EQ(match(pool, *literal_close, ")"), MatchResult::Match);
  EXPECT_EQ(match(pool, *literal_close, "x"), MatchResult::NoMatch);
}

TEST(LlvmLibcRegexMatcherTest, BracketMatchingAndRanges) {
  ExprPool pool;
  ErrorOr<ExprId> range = parse_ere("x[c-f]y", REG_EXTENDED, pool);
  ASSERT_TRUE(range.has_value());
  EXPECT_EQ(match(pool, *range, "xcy"), MatchResult::Match);
  EXPECT_EQ(match(pool, *range, "xdy"), MatchResult::Match);
  EXPECT_EQ(match(pool, *range, "xey"), MatchResult::Match);
  EXPECT_EQ(match(pool, *range, "xfy"), MatchResult::Match);
  EXPECT_EQ(match(pool, *range, "xby"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *range, "xgy"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *range, "xy"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *range, "xccy"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *range, "x-y"), MatchResult::NoMatch);

  ErrorOr<ExprId> negated = parse_ere("x[^c-f]y", REG_EXTENDED, pool);
  ASSERT_TRUE(negated.has_value());
  EXPECT_EQ(match(pool, *negated, "xby"), MatchResult::Match);
  EXPECT_EQ(match(pool, *negated, "xgy"), MatchResult::Match);
  EXPECT_EQ(match(pool, *negated, "xcy"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *negated, "xfy"), MatchResult::NoMatch);
  EXPECT_EQ(match(pool, *negated, "xy"), MatchResult::NoMatch);

  ErrorOr<ExprId> nl_neg =
      parse_ere("x[^c-f]y", REG_EXTENDED | REG_NEWLINE, pool);
  ASSERT_TRUE(nl_neg.has_value());
  EXPECT_EQ(match(pool, *nl_neg, "x\ny", REG_NEWLINE, 0), MatchResult::NoMatch);

  ErrorOr<ExprId> first_rb = parse_ere("a[]b]c", REG_EXTENDED, pool);
  ASSERT_TRUE(first_rb.has_value());
  EXPECT_EQ(match(pool, *first_rb, "a]c"), MatchResult::Match);
  EXPECT_EQ(match(pool, *first_rb, "abc"), MatchResult::Match);
  EXPECT_EQ(match(pool, *first_rb, "ac"), MatchResult::NoMatch);

  ErrorOr<ExprId> lead_dash = parse_ere("a[-b]c", REG_EXTENDED, pool);
  ASSERT_TRUE(lead_dash.has_value());
  EXPECT_EQ(match(pool, *lead_dash, "a-c"), MatchResult::Match);
  EXPECT_EQ(match(pool, *lead_dash, "abc"), MatchResult::Match);

  ErrorOr<ExprId> trail_dash = parse_ere("a[b-]c", REG_EXTENDED, pool);
  ASSERT_TRUE(trail_dash.has_value());
  EXPECT_EQ(match(pool, *trail_dash, "a-c"), MatchResult::Match);
  EXPECT_EQ(match(pool, *trail_dash, "abc"), MatchResult::Match);

  ErrorOr<ExprId> dash_start_range = parse_ere("a[--0]c", REG_EXTENDED, pool);
  ASSERT_TRUE(dash_start_range.has_value());
  EXPECT_EQ(match(pool, *dash_start_range, "a-c"), MatchResult::Match);
  EXPECT_EQ(match(pool, *dash_start_range, "a.c"), MatchResult::Match);
  EXPECT_EQ(match(pool, *dash_start_range, "a/c"), MatchResult::Match);
  EXPECT_EQ(match(pool, *dash_start_range, "a0c"), MatchResult::Match);
  EXPECT_EQ(match(pool, *dash_start_range, "a1c"), MatchResult::NoMatch);

  ErrorOr<ExprId> dash_end_range = parse_ere("a[!--]c", REG_EXTENDED, pool);
  ASSERT_TRUE(dash_end_range.has_value());
  EXPECT_EQ(match(pool, *dash_end_range, "a!c"), MatchResult::Match);
  EXPECT_EQ(match(pool, *dash_end_range, "a+c"), MatchResult::Match);
  EXPECT_EQ(match(pool, *dash_end_range, "a-c"), MatchResult::Match);
  EXPECT_EQ(match(pool, *dash_end_range, "a.c"), MatchResult::NoMatch);

  ErrorOr<ExprId> coll_sym = parse_ere("a[[.x.]]b", REG_EXTENDED, pool);
  ASSERT_TRUE(coll_sym.has_value());
  EXPECT_EQ(match(pool, *coll_sym, "axb"), MatchResult::Match);

  ErrorOr<ExprId> coll_rb = parse_ere("a[[.].]]b", REG_EXTENDED, pool);
  ASSERT_TRUE(coll_rb.has_value());
  EXPECT_EQ(match(pool, *coll_rb, "a]b"), MatchResult::Match);

  ErrorOr<ExprId> coll_one = parse_ere("a[[.one.]]b", REG_EXTENDED, pool);
  ASSERT_TRUE(coll_one.has_value());
  EXPECT_EQ(match(pool, *coll_one, "a1b"), MatchResult::Match);

  ErrorOr<ExprId> star_range = parse_ere("^[a-c]*$", REG_EXTENDED, pool);
  ASSERT_TRUE(star_range.has_value());
  EXPECT_EQ(match(pool, *star_range, ""), MatchResult::Match);
  EXPECT_EQ(match(pool, *star_range, "abcba"), MatchResult::Match);
  EXPECT_EQ(match(pool, *star_range, "abzcba"), MatchResult::NoMatch);

  ErrorOr<ExprId> alpha = parse_ere("^[[:alpha:]]*$", REG_EXTENDED, pool);
  ASSERT_TRUE(alpha.has_value());
  EXPECT_EQ(match(pool, *alpha, ""), MatchResult::Match);
  EXPECT_EQ(match(pool, *alpha, "HelloWorld"), MatchResult::Match);
  EXPECT_EQ(match(pool, *alpha, "Hello1World"), MatchResult::NoMatch);

  ErrorOr<ExprId> sh_w = parse_ere(R"(^\w*$)", REG_EXTENDED, pool);
  ASSERT_TRUE(sh_w.has_value());
  EXPECT_EQ(match(pool, *sh_w, ""), MatchResult::Match);
  EXPECT_EQ(match(pool, *sh_w, "Hello_123"), MatchResult::Match);
  EXPECT_EQ(match(pool, *sh_w, "Hello-123"), MatchResult::NoMatch);

  ErrorOr<ExprId> sh_d = parse_ere(R"(^\d*$)", REG_EXTENDED, pool);
  ASSERT_TRUE(sh_d.has_value());
  EXPECT_EQ(match(pool, *sh_d, ""), MatchResult::Match);
  EXPECT_EQ(match(pool, *sh_d, "12345"), MatchResult::Match);
  EXPECT_EQ(match(pool, *sh_d, "123a45"), MatchResult::NoMatch);

  ErrorOr<ExprId> sh_neg = parse_ere(R"(x\Wy)", REG_EXTENDED, pool);
  ASSERT_TRUE(sh_neg.has_value());
  EXPECT_EQ(match(pool, *sh_neg, "x y"), MatchResult::Match);
  EXPECT_EQ(match(pool, *sh_neg, "x_y"), MatchResult::NoMatch);

  ErrorOr<ExprId> sh_neg_nl =
      parse_ere(R"(x\Wy)", REG_EXTENDED | REG_NEWLINE, pool);
  ASSERT_TRUE(sh_neg_nl.has_value());
  EXPECT_EQ(match(pool, *sh_neg_nl, "x\ny", REG_NEWLINE, 0),
            MatchResult::NoMatch);

  ErrorOr<ExprId> multi_class =
      parse_ere("^[[:alpha:][:blank:]]*$", REG_EXTENDED, pool);
  ASSERT_TRUE(multi_class.has_value());
  EXPECT_EQ(match(pool, *multi_class, "Hello World"), MatchResult::Match);
  EXPECT_EQ(match(pool, *multi_class, "Hello1World"), MatchResult::NoMatch);
}

TEST(LlvmLibcRegexMatcherTest, PoolExhaustionPropagation) {
  ExprPool pool;
  const ExprId empty_str = *pool.empty_str();
  const ExprId lit_a = *pool.make_lit('a');
  const ExprId lit_b = *pool.make_lit('b');
  const ExprId lit_c = *pool.make_lit('c');

  const ExprId ab = *pool.make_concat(lit_a, lit_b);
  const ExprId ac = *pool.make_concat(lit_a, lit_c);
  const ExprId opt_a = *pool.make_alt(lit_a, empty_str);
  // D_a((a . b) | (a . c)) = b | c, and D_a((a . b)*) = b . (a . b)*.
  const ExprId need_bc = *pool.make_alt(ab, ac);
  const ExprId star_ab = *pool.make_star(ab);
  const ExprId concat_left_fail = *pool.make_concat(need_bc, lit_c);
  const ExprId concat_right_fail = *pool.make_concat(opt_a, need_bc);
  const ExprId alt_right_fail = *pool.make_alt(lit_a, need_bc);
  const ExprId lit_d = *pool.make_lit('d');
  const ExprId alt_left_fail = *pool.make_alt(need_bc, lit_d);

  // Fill the remainder of the pool to MAX_NODE_LIMIT (10000 nodes).
  ExprId chain = lit_a;
  while (pool.size() < 10000)
    chain = *pool.make_concat(lit_a, chain);

  const ExprId exhausted_exprs[] = {need_bc,          star_ab,
                                    concat_left_fail, concat_right_fail,
                                    alt_left_fail,    alt_right_fail};
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

TEST(LlvmLibcRegexMatcherTest, AlternationAndGroupingStabilization) {
  ExprPool pool;
  ErrorOr<ExprId> ere =
      parse_ere("(wee|week)(knights|night)", REG_EXTENDED, pool);
  ASSERT_TRUE(ere.has_value());
  EXPECT_EQ(match(pool, *ere, "weeknights"), MatchResult::Match);
  EXPECT_EQ(match(pool, *ere, "weenight"), MatchResult::Match);
  EXPECT_EQ(match(pool, *ere, "weeknight"), MatchResult::Match);
  EXPECT_EQ(match(pool, *ere, "weeknightsxyz"), MatchResult::Match);
  EXPECT_EQ(match(pool, *ere, "weeks"), MatchResult::NoMatch);

  // Derivative state pool stabilizes after warming up on matches.
  size_t warm_nodes = pool.size();
  EXPECT_EQ(match(pool, *ere, "weeknights"), MatchResult::Match);
  EXPECT_EQ(match(pool, *ere, "weenight"), MatchResult::Match);
  EXPECT_EQ(pool.size(), warm_nodes);

  // Parentheses control precedence without adding Group wrapper nodes,
  // preserving hash-consing deduplication.
  ErrorOr<ExprId> lit_a = parse_ere("a", REG_EXTENDED, pool);
  ErrorOr<ExprId> grouped_a = parse_ere("((a))", REG_EXTENDED, pool);
  ASSERT_TRUE(lit_a.has_value());
  ASSERT_TRUE(grouped_a.has_value());
  EXPECT_EQ(*grouped_a, *lit_a);

  ErrorOr<ExprId> ab = parse_ere("a|b", REG_EXTENDED, pool);
  ErrorOr<ExprId> grouped_ab = parse_ere("(a|b)|(b|a)", REG_EXTENDED, pool);
  ASSERT_TRUE(ab.has_value());
  ASSERT_TRUE(grouped_ab.has_value());
  EXPECT_EQ(*grouped_ab, *ab);

  // Right-to-left alternation folding interns only the 2 base nodes, 4 literal
  // nodes, and 3 Alt spine nodes (9 total) with zero orphaned Alt nodes.
  ExprPool alt_pool;
  ErrorOr<ExprId> multi_alt = parse_ere("a|b|c|d", REG_EXTENDED, alt_pool);
  ASSERT_TRUE(multi_alt.has_value());
  EXPECT_EQ(alt_pool.size(), size_t(9));
}
