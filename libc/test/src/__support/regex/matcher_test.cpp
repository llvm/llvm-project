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

  const string_view bad_rpt_patterns[] = {
      "*",     "+",     "?",     "{1}",     "{1",    "^*",    "^+",
      "^?",    "^{1}",  "$*",    "${1}",    "a**",   "a*+",   "a*?",
      "a*{1}", "a+{1}", "a?{1}", "a{2}{2}", "a{2}*", "a{2}+", "a{2}?"};
  for (string_view bad_rpt : bad_rpt_patterns) {
    ErrorOr<ExprId> err = parse_ere(bad_rpt, REG_EXTENDED, pool);
    ASSERT_FALSE(err.has_value());
    EXPECT_EQ(err.error(), REG_BADRPT);
  }

  const string_view ebrace_patterns[] = {"a{1", "a{1a", "a{1,2", "a{1,x"};
  for (string_view pat : ebrace_patterns) {
    ErrorOr<ExprId> err = parse_ere(pat, REG_EXTENDED, pool);
    ASSERT_FALSE(err.has_value());
    EXPECT_EQ(err.error(), REG_EBRACE);
  }

  const string_view badbr_patterns[] = {"a{1a}",    "a{1,x}",       "a{1,0}",
                                        "a{5,2}",   "a{256}",       "a{1,256}",
                                        "a{1,2,3}", "a{9876543210}"};
  for (string_view pat : badbr_patterns) {
    ErrorOr<ExprId> err = parse_ere(pat, REG_EXTENDED, pool);
    ASSERT_FALSE(err.has_value());
    EXPECT_EQ(err.error(), REG_BADBR);
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
  const ExprId rep_ab = *pool.make_repeat(ab, 2, 3);
  const ExprId concat_left_fail = *pool.make_concat(need_bc, lit_c);
  const ExprId concat_right_fail = *pool.make_concat(opt_a, need_bc);
  const ExprId alt_right_fail = *pool.make_alt(lit_a, need_bc);
  const ExprId lit_d = *pool.make_lit('d');
  const ExprId alt_left_fail = *pool.make_alt(need_bc, lit_d);

  // Fill the remainder of the pool to MAX_NODE_LIMIT (10000 nodes).
  ExprId chain = lit_a;
  while (pool.size() < 10000)
    chain = *pool.make_concat(lit_a, chain);

  const ExprId exhausted_exprs[] = {
      need_bc,           star_ab,       rep_ab,        concat_left_fail,
      concat_right_fail, alt_left_fail, alt_right_fail};
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

TEST(LlvmLibcRegexMatcherTest, PlusAndQuestionDesugaringAndStabilization) {
  ExprPool pool;
  // r+ pushes r and Star(r) separately onto ParsedAtomStack and r? lowers to
  // Alt(EmptyStr, r), interning to identical ExprId handles without orphaned
  // Concat spines.
  ErrorOr<ExprId> plus_a = parse_ere("a+", REG_EXTENDED, pool);
  ErrorOr<ExprId> concat_star_a = parse_ere("aa*", REG_EXTENDED, pool);
  ASSERT_TRUE(plus_a.has_value());
  ASSERT_TRUE(concat_star_a.has_value());
  EXPECT_EQ(*plus_a, *concat_star_a);

  ErrorOr<ExprId> qmark_a = parse_ere("a?", REG_EXTENDED, pool);
  ErrorOr<ExprId> alt_empty_a = parse_ere("()|a", REG_EXTENDED, pool);
  ASSERT_TRUE(qmark_a.has_value());
  ASSERT_TRUE(alt_empty_a.has_value());
  EXPECT_EQ(*qmark_a, *alt_empty_a);

  ErrorOr<ExprId> ere = parse_ere("a+b?c", REG_EXTENDED, pool);
  ASSERT_TRUE(ere.has_value());
  EXPECT_EQ(match(pool, *ere, "aaabc"), MatchResult::Match);
  size_t warm_nodes = pool.size();
  EXPECT_EQ(match(pool, *ere, "aaabc"), MatchResult::Match);
  EXPECT_EQ(pool.size(), warm_nodes);

  // Pushing r and Star(r) as separate stack entries builds r . (r* . rest)
  // once, giving (ab)+c and (ab)(ab)*c identical 10-node pools and allowing
  // ^(a...2500_as...)+$ to compile in 5,007 nodes and match 2,500 'a's.
  ExprPool plus_pool;
  ExprPool star_pool;
  ErrorOr<ExprId> group_plus = parse_ere("(ab)+c", REG_EXTENDED, plus_pool);
  ErrorOr<ExprId> group_star = parse_ere("(ab)(ab)*c", REG_EXTENDED, star_pool);
  ASSERT_TRUE(group_plus.has_value());
  ASSERT_TRUE(group_star.has_value());
  EXPECT_EQ(plus_pool.size(), size_t(10));
  EXPECT_EQ(plus_pool.size(), star_pool.size());

  char pat2500[2506];
  pat2500[0] = '^';
  pat2500[1] = '(';
  for (size_t i = 0; i < 2500; ++i)
    pat2500[2 + i] = 'a';
  pat2500[2502] = ')';
  pat2500[2503] = '+';
  pat2500[2504] = '$';
  pat2500[2505] = '\0';
  ExprPool long_plus_pool;
  ErrorOr<ExprId> long_plus =
      parse_ere(string_view(pat2500, 2505), REG_EXTENDED, long_plus_pool);
  ASSERT_TRUE(long_plus.has_value());
  EXPECT_EQ(long_plus_pool.size(), size_t(5007));
  EXPECT_EQ(match(long_plus_pool, *long_plus, string_view(pat2500 + 2, 2500)),
            MatchResult::Match);
}

TEST(LlvmLibcRegexMatcherTest, RepeatStabilizationAndRfc2822Grammar) {
  // Native Repeat(body, 255, 255) compiles into 4 nodes inside 1 KiB; both
  // unanchored a{255} / a{255}b (via make_alt interval coalescing) and anchored
  // ^a{255}$ match and stabilize on warm runs.
  ExprPool rep_pool;
  ErrorOr<ExprId> a255 = parse_ere("a{255}", REG_EXTENDED, rep_pool);
  ASSERT_TRUE(a255.has_value());
  EXPECT_EQ(rep_pool.size(), size_t(4));
  EXPECT_LE(rep_pool.allocated_bytes(), size_t(1024));

  ErrorOr<ExprId> a255b = parse_ere("a{255}b", REG_EXTENDED, rep_pool);
  ErrorOr<ExprId> anchored_a255 = parse_ere("^a{255}$", REG_EXTENDED, rep_pool);
  ASSERT_TRUE(a255b.has_value());
  ASSERT_TRUE(anchored_a255.has_value());
  char buf256[256];
  for (char &ch : buf256)
    ch = 'a';
  EXPECT_EQ(match(rep_pool, *a255, string_view(buf256, 255)),
            MatchResult::Match);
  buf256[255] = 'b';
  EXPECT_EQ(match(rep_pool, *a255b, string_view(buf256, 256)),
            MatchResult::Match);
  buf256[255] = 'a';
  EXPECT_EQ(match(rep_pool, *anchored_a255, string_view(buf256, 255)),
            MatchResult::Match);
  size_t warm_nodes = rep_pool.size();
  EXPECT_EQ(match(rep_pool, *a255, string_view(buf256, 255)),
            MatchResult::Match);
  EXPECT_EQ(match(rep_pool, *anchored_a255, string_view(buf256, 255)),
            MatchResult::Match);
  EXPECT_EQ(rep_pool.size(), warm_nodes);
  EXPECT_EQ(match(rep_pool, *anchored_a255, string_view(buf256, 254)),
            MatchResult::NoMatch);
  EXPECT_EQ(match(rep_pool, *anchored_a255, string_view(buf256, 256)),
            MatchResult::NoMatch);

  // Shared-prefix Concat derivatives (^(a+){255}$, (ab?){255}b) and interleaved
  // multi-branch spines (x?.{255}a|.{255}b) coalesce without hitting
  // MAX_NODE_LIMIT, and cached Expr::rep_expansion parses 30-deep nested '+'
  // inside '{2}' in O(1) per node.
  ExprPool prefix_pool;
  ErrorOr<ExprId> plus_255 =
      parse_ere("^(a+){255}$", REG_EXTENDED, prefix_pool);
  ErrorOr<ExprId> opt_255b =
      parse_ere("(ab?){255}b", REG_EXTENDED, prefix_pool);
  ErrorOr<ExprId> inter_255 =
      parse_ere("x?.{255}a|.{255}b", REG_EXTENDED, prefix_pool);
  ASSERT_TRUE(plus_255.has_value());
  ASSERT_TRUE(opt_255b.has_value());
  ASSERT_TRUE(inter_255.has_value());
  EXPECT_EQ(match(prefix_pool, *plus_255, string_view(buf256, 255)),
            MatchResult::Match);
  buf256[255] = 'b';
  EXPECT_EQ(match(prefix_pool, *opt_255b, string_view(buf256, 256)),
            MatchResult::Match);
  EXPECT_EQ(match(prefix_pool, *inter_255, string_view(buf256, 256)),
            MatchResult::Match);

  ExprPool dag_exp_pool;
  ErrorOr<ExprId> deep_plus_rep = parse_ere(
      "(((((((((((((((((((((((((((((((a)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+)+"
      ")+)+)+)+)+)+)+)+)+)+)+)+)+)+){2}",
      REG_EXTENDED, dag_exp_pool);
  ASSERT_TRUE(deep_plus_rep.has_value());
  EXPECT_EQ(match(dag_exp_pool, *deep_plus_rep, "aa"), MatchResult::Match);

  // Build a >7 KiB ERE from RFC-2822 Section 3.4.1 addr-spec EBNF productions
  // (F = FWS, C = CFWS, T = atext, W = word, S = addr-spec) and verify
  // hash-consed DAG deduplication keeps the pool under 32 KiB.
  char rfc_buf[16384];
  size_t rfc_len = 0;
  auto expand = [&](auto &self, string_view str) -> void {
    for (char ch : str) {
      if (ch == 'F') {
        self(self, "(([ \t]*\r\n)?[ \t]+)");
      } else if (ch == 'C') {
        self(self, R"(((F?\((F?([!-'*-[^-~]|\\([ -~]|[ )"
                   "\t"
                   R"(])))*F?\))+F?|F)?)");
      } else if (ch == 'T') {
        self(self, R"([[:alnum:]!#$%&'*+/=?^_`{|}~-])");
      } else if (ch == 'W') {
        self(self, R"(((CT+C)|(C"(F?([!#-[^-~]|\\([ -~]|[ )"
                   "\t"
                   R"(])))*F?"C)))");
      } else if (ch == 'S') {
        self(self, R"((((CT+(\.T+)*C)|W|(W(\.W{1,1}){0,31}))@)"
                   R"(((CT+(\.T+)*C)|(C\[(([0-9]{1,3}\.){3}[0-9]{1,3}|)"
                   R"((F?([!-Z^-~]|\\([ -~]|[ )"
                   "\t"
                   R"(])))*F?)\]C)|)"
                   R"(((CT+C)(\.(CT+C){1,1}){1,31}))))");
      } else if (rfc_len < sizeof(rfc_buf)) {
        rfc_buf[rfc_len] = ch;
        ++rfc_len;
      }
    }
  };
  expand(expand, "^(S|W{0,8}<S>)$");
  string_view rfc2822_ere(rfc_buf, rfc_len);
  EXPECT_GT(rfc2822_ere.size(), size_t(7000));

  ExprPool rfc_pool;
  ErrorOr<ExprId> addr_spec = parse_ere(rfc2822_ere, REG_EXTENDED, rfc_pool);
  ASSERT_TRUE(addr_spec.has_value());
  EXPECT_LE(rfc_pool.allocated_bytes(), size_t(32768));

  const string_view good_addrs[] = {"user.name+tag@example.com",
                                    R"("quoted..user"@[192.168.1.1])",
                                    "(comment) alice@sub.domain.org (ok)",
                                    "(\r\n folded comment )alice@example.com",
                                    "Alice Smith <alice@example.com>"};
  for (string_view addr : good_addrs)
    EXPECT_EQ(match(rfc_pool, *addr_spec, addr), MatchResult::Match);
  const string_view bad_addrs[] = {"user..name@example.com", "@example.com",
                                   "user@", "user@.example.com",
                                   "user@example..com"};
  for (string_view addr : bad_addrs)
    EXPECT_EQ(match(rfc_pool, *addr_spec, addr), MatchResult::NoMatch);
}
