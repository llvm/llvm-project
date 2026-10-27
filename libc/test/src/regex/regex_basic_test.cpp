//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Basic round-trip tests for POSIX regex functions.
///
//===----------------------------------------------------------------------===//

#include "src/regex/regcomp.h"
#include "src/regex/regexec.h"
#include "src/regex/regfree.h"
#include "test/UnitTest/Test.h"

#include "hdr/regex_macros.h"
#include "hdr/types/regex_t.h"
#include "src/__support/CPP/scope.h"

using LIBC_NAMESPACE::cpp::scope_exit;

TEST(LlvmLibcRegexTest, BasicLiteralRoundTrip) {
  regex_t preg;
  ASSERT_EQ(0,
            LIBC_NAMESPACE::regcomp(&preg, "hello", REG_EXTENDED | REG_NOSUB));
  scope_exit free_guard([&preg] { LIBC_NAMESPACE::regfree(&preg); });
  EXPECT_EQ(0,
            LIBC_NAMESPACE::regexec(&preg, "say hello world", 0, nullptr, 0));
  EXPECT_EQ(REG_NOMATCH,
            LIBC_NAMESPACE::regexec(&preg, "goodbye", 0, nullptr, 0));
}

TEST(LlvmLibcRegexTest, MismatchCases) {
  regex_t preg;
  // Partial match
  ASSERT_EQ(0,
            LIBC_NAMESPACE::regcomp(&preg, "hello", REG_EXTENDED | REG_NOSUB));
  EXPECT_EQ(REG_NOMATCH, LIBC_NAMESPACE::regexec(&preg, "hell", 0, nullptr, 0));
  LIBC_NAMESPACE::regfree(&preg);

  // Case sensitivity
  ASSERT_EQ(0,
            LIBC_NAMESPACE::regcomp(&preg, "Hello", REG_EXTENDED | REG_NOSUB));
  EXPECT_EQ(REG_NOMATCH,
            LIBC_NAMESPACE::regexec(&preg, "hello", 0, nullptr, 0));
  LIBC_NAMESPACE::regfree(&preg);

  // Empty string vs non-empty pattern
  ASSERT_EQ(0, LIBC_NAMESPACE::regcomp(&preg, "a", REG_EXTENDED | REG_NOSUB));
  EXPECT_EQ(REG_NOMATCH, LIBC_NAMESPACE::regexec(&preg, "", 0, nullptr, 0));
  LIBC_NAMESPACE::regfree(&preg);
}

TEST(LlvmLibcRegexTest, EmptyString) {
  regex_t preg;
  ASSERT_EQ(0, LIBC_NAMESPACE::regcomp(&preg, "", REG_EXTENDED | REG_NOSUB));
  scope_exit free_guard([&preg] { LIBC_NAMESPACE::regfree(&preg); });
  EXPECT_EQ(0, LIBC_NAMESPACE::regexec(&preg, "anything", 0, nullptr, 0));
  EXPECT_EQ(0, LIBC_NAMESPACE::regexec(&preg, "", 0, nullptr, 0));
}

TEST(LlvmLibcRegexTest, ExactMatch) {
  regex_t preg;
  ASSERT_EQ(0,
            LIBC_NAMESPACE::regcomp(&preg, "test", REG_EXTENDED | REG_NOSUB));
  scope_exit free_guard([&preg] { LIBC_NAMESPACE::regfree(&preg); });
  EXPECT_EQ(0, LIBC_NAMESPACE::regexec(&preg, "test", 0, nullptr, 0));
}

TEST(LlvmLibcRegexTest, NullByteStopsParsing) {
  regex_t preg;
  ASSERT_EQ(0,
            LIBC_NAMESPACE::regcomp(&preg, "match", REG_EXTENDED | REG_NOSUB));
  scope_exit free_guard([&preg] { LIBC_NAMESPACE::regfree(&preg); });
  EXPECT_EQ(REG_NOMATCH,
            LIBC_NAMESPACE::regexec(&preg, "doesn't \0 match", 0, nullptr, 0));
}

TEST(LlvmLibcRegexTest, OverlappingPrefixAndPostFreeSafety) {
  regex_t preg;
  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "ababac", REG_EXTENDED | REG_NOSUB),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abababac", 0, nullptr, 0), 0);
  LIBC_NAMESPACE::regfree(&preg);
  // Repeated regfree and post-free regexec must be safe.
  LIBC_NAMESPACE::regfree(&preg);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abababac", 0, nullptr, 0),
            REG_NOMATCH);
}

TEST(LlvmLibcRegexTest, PoolExhaustionReturnsEspace) {
  // 9998 'a' bytes plus EmptySet (1) and EmptyStr (2) fill all 10000 ExprPool
  // slots during regcomp; regexec then hits MAX_NODE_LIMIT when interning the
  // first derivative Alt state.
  char pattern[10000];
  for (size_t i = 0; i < 9998; ++i)
    pattern[i] = 'a';
  pattern[9998] = '\0';

  regex_t preg;
  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, pattern, REG_EXTENDED | REG_NOSUB),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a", 0, nullptr, 0), REG_ESPACE);
  LIBC_NAMESPACE::regfree(&preg);

  // One additional byte (9999 'a's) hits MAX_NODE_LIMIT during regcomp;
  // calling regfree afterwards must be a safe no-op.
  pattern[9998] = 'a';
  pattern[9999] = '\0';
  EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, pattern, REG_EXTENDED | REG_NOSUB),
            REG_ESPACE);
  LIBC_NAMESPACE::regfree(&preg);
}

TEST(LlvmLibcRegexTest, WildcardDot) {
  regex_t preg;
  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "a.c", REG_EXTENDED | REG_NOSUB), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abc", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "axc", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "ac", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abbc", 0, nullptr, 0), REG_NOMATCH);
  LIBC_NAMESPACE::regfree(&preg);

  ASSERT_EQ(
      LIBC_NAMESPACE::regcomp(&preg, R"(a\.b.c)", REG_EXTENDED | REG_NOSUB), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a.bXc", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "aXbXc", 0, nullptr, 0),
            REG_NOMATCH);
  LIBC_NAMESPACE::regfree(&preg);
}

TEST(LlvmLibcRegexTest, AnchorsAndFlags) {
  regex_t preg;
  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "^foo",
                                    REG_EXTENDED | REG_NOSUB | REG_NEWLINE),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "foobar", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xfoo", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "foo", 0, nullptr, REG_NOTBOL),
            REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "bar\nfoo", 0, nullptr, REG_NOTBOL),
            0);
  LIBC_NAMESPACE::regfree(&preg);

  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "foo$",
                                    REG_EXTENDED | REG_NOSUB | REG_NEWLINE),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "barfoo", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "foox", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "foo", 0, nullptr, REG_NOTEOL),
            REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "foo\nbar", 0, nullptr, REG_NOTEOL),
            0);
  LIBC_NAMESPACE::regfree(&preg);

  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "^$",
                                    REG_EXTENDED | REG_NOSUB | REG_NEWLINE),
            0);
  scope_exit free_guard([&preg] { LIBC_NAMESPACE::regfree(&preg); });
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a\n\nb", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a\nb", 0, nullptr, 0), REG_NOMATCH);
}

TEST(LlvmLibcRegexTest, KleeneStarAndBadRpt) {
  regex_t preg;
  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "ab*c", REG_EXTENDED | REG_NOSUB),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "ac", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abc", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abbbc", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abbd", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "bbbc", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abXc", 0, nullptr, 0), REG_NOMATCH);
  LIBC_NAMESPACE::regfree(&preg);

  const char *const bad_rpt[] = {"*", "+", "?", "{1}", "^*", "$*", "a**"};
  for (const char *pat : bad_rpt) {
    EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, pat, REG_EXTENDED), REG_BADRPT);
    LIBC_NAMESPACE::regfree(&preg);
  }
}

TEST(LlvmLibcRegexTest, EscapedMetacharacters) {
  regex_t preg;
  ASSERT_EQ(
      LIBC_NAMESPACE::regcomp(&preg, R"(a\.b\*c\\d)", REG_EXTENDED | REG_NOSUB),
      0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, R"(xx a.b*c\d yy)", 0, nullptr, 0),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, R"(aXb*c\d)", 0, nullptr, 0),
            REG_NOMATCH);
  LIBC_NAMESPACE::regfree(&preg);

  EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, R"(trailing\)", REG_EXTENDED),
            REG_EESCAPE);
  LIBC_NAMESPACE::regfree(&preg);
}

TEST(LlvmLibcRegexTest, BracketExpressionsAndRanges) {
  regex_t preg;
  // Matching and range checks with near-miss boundary tests.
  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "x[c-f]y", REG_EXTENDED | REG_NOSUB),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xcy", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xdy", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xey", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xfy", 0, nullptr, 0), 0);
  // Near-misses: adjacent ASCII values 'b' (c - 1) and 'g' (f + 1), empty,
  // multi-char, and literal '-'.
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xby", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xgy", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xy", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xccy", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "x-y", 0, nullptr, 0), REG_NOMATCH);
  LIBC_NAMESPACE::regfree(&preg);

  // Negated brackets and REG_NEWLINE interaction.
  ASSERT_EQ(
      LIBC_NAMESPACE::regcomp(&preg, "x[^c-f]y", REG_EXTENDED | REG_NOSUB), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xby", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xgy", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xcy", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xfy", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xy", 0, nullptr, 0), REG_NOMATCH);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "xbbby", 0, nullptr, 0),
            REG_NOMATCH);
  LIBC_NAMESPACE::regfree(&preg);

  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "x[^c-f]y",
                                    REG_EXTENDED | REG_NEWLINE | REG_NOSUB),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "x\ny", 0, nullptr, 0), REG_NOMATCH);
  LIBC_NAMESPACE::regfree(&preg);

  // First-position ']', leading '-', trailing '-', and collating elements.
  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "a[]b]c", REG_EXTENDED | REG_NOSUB),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a]c", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abc", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "ac", 0, nullptr, 0), REG_NOMATCH);
  LIBC_NAMESPACE::regfree(&preg);

  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "a[-b]c", REG_EXTENDED | REG_NOSUB),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a-c", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abc", 0, nullptr, 0), 0);
  LIBC_NAMESPACE::regfree(&preg);

  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "a[b-]c", REG_EXTENDED | REG_NOSUB),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a-c", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abc", 0, nullptr, 0), 0);
  LIBC_NAMESPACE::regfree(&preg);

  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "a[--0]c", REG_EXTENDED | REG_NOSUB),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a-c", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a.c", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a/c", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a0c", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a1c", 0, nullptr, 0), REG_NOMATCH);
  LIBC_NAMESPACE::regfree(&preg);

  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "a[!--]c", REG_EXTENDED | REG_NOSUB),
            0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a!c", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a+c", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a-c", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a.c", 0, nullptr, 0), REG_NOMATCH);
  LIBC_NAMESPACE::regfree(&preg);

  ASSERT_EQ(
      LIBC_NAMESPACE::regcomp(&preg, "a[[.x.]]b", REG_EXTENDED | REG_NOSUB), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "axb", 0, nullptr, 0), 0);
  LIBC_NAMESPACE::regfree(&preg);

  ASSERT_EQ(
      LIBC_NAMESPACE::regcomp(&preg, "a[[.].]]b", REG_EXTENDED | REG_NOSUB), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a]b", 0, nullptr, 0), 0);
  LIBC_NAMESPACE::regfree(&preg);

  ASSERT_EQ(
      LIBC_NAMESPACE::regcomp(&preg, "^[a-c]*$", REG_EXTENDED | REG_NOSUB), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abcba", 0, nullptr, 0), 0);
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "abzcba", 0, nullptr, 0),
            REG_NOMATCH);
  LIBC_NAMESPACE::regfree(&preg);

  // Bracket error conditions.
  const char *const ebrack_patterns[] = {
      "[", "[^", "[abc", "[]", "a[[.x", "a[[.x.", "a[[:digit:", "a[[:digit"};
  for (const char *pat : ebrack_patterns) {
    EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, pat, REG_EXTENDED), REG_EBRACK);
    LIBC_NAMESPACE::regfree(&preg);
  }

  const char *const erange_patterns[] = {"a[3-1]c", "a[1-3-5]c", "a[1-",
                                         "a[a-[:digit:]]c", "a[[:alpha:]-z]c"};
  for (const char *pat : erange_patterns) {
    EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, pat, REG_EXTENDED), REG_ERANGE);
    LIBC_NAMESPACE::regfree(&preg);
  }

  const char *const ecollate_patterns[] = {"a[[.notdef.]]b", "a[[.x,.]]",
                                           "a[[=b,=]]"};
  for (const char *pat : ecollate_patterns) {
    EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, pat, REG_EXTENDED), REG_ECOLLATE);
    LIBC_NAMESPACE::regfree(&preg);
  }

  const char *const ectype_patterns[] = {"a[[:notdef:]]b", "a[[:alph:]]b",
                                         "a[[:]:]]b", "a[[:-:]]b"};
  for (const char *pat : ectype_patterns) {
    EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, pat, REG_EXTENDED), REG_ECTYPE);
    LIBC_NAMESPACE::regfree(&preg);
  }
}

TEST(LlvmLibcRegexTest, UnimplementedSyntaxAndFlags) {
  regex_t preg;
  // Missing REG_EXTENDED (BRE mode) and unsupported cflags return REG_BADPAT.
  EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, "a", 0), REG_BADPAT);
  LIBC_NAMESPACE::regfree(&preg);
  EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, "a", REG_EXTENDED | REG_ICASE),
            REG_BADPAT);
  LIBC_NAMESPACE::regfree(&preg);
  EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, "a", ~(REG_NOSUB | REG_NEWLINE)),
            REG_BADPAT);
  LIBC_NAMESPACE::regfree(&preg);

  // Unimplemented ERE metacharacters and special escapes return REG_BADPAT.
  const char *const bad_patterns[] = {
      "a+", "a?", "(a)", "a{1}", "a|b", R"(\1)", R"(\b)", "[[:<:]]", "[[:>:]]"};
  for (const char *pat : bad_patterns) {
    EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, pat, REG_EXTENDED), REG_BADPAT);
    LIBC_NAMESPACE::regfree(&preg);
  }

  // Unsupported eflags in regexec return REG_BADPAT.
  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "a", REG_EXTENDED), 0);
  scope_exit free_guard([&preg] { LIBC_NAMESPACE::regfree(&preg); });
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "a", 0, nullptr,
                                    ~(REG_NOTBOL | REG_NOTEOL)),
            REG_BADPAT);
}

TEST(LlvmLibcRegexTest, PosixCharacterClassesAndShorthands) {
  auto check = [](const char *pat, const char *str, int expected = 0) {
    regex_t preg;
    ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, pat, REG_EXTENDED | REG_NOSUB), 0);
    EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, str, 0, nullptr, 0), expected);
    LIBC_NAMESPACE::regfree(&preg);
  };

  // Near-miss ASCII neighbours for [:digit:] and \d.
  check("^x[[:digit:]]y$", "x0y");
  check("^x[[:digit:]]y$", "x9y");
  check("^x[[:digit:]]y$", "x/y", REG_NOMATCH);
  check("^x[[:digit:]]y$", "x:y", REG_NOMATCH);
  check("^x[[:digit:]]y$", "x12y", REG_NOMATCH);

  check(R"(^x\dy$)", "x5y");
  check(R"(^x\dy$)", "x/y", REG_NOMATCH);
  check(R"(^x\dy$)", "x:y", REG_NOMATCH);

  // Near-miss ASCII neighbours for [:xdigit:].
  check("^x[[:xdigit:]]y$", "xfy");
  check("^x[[:xdigit:]]y$", "xFy");
  check("^x[[:xdigit:]]y$", "xgy", REG_NOMATCH);
  check("^x[[:xdigit:]]y$", "xGy", REG_NOMATCH);

  // [:blank:] matches only space and tab, not newline.
  check("^x[[:blank:]]y$", "x y");
  check("^x[[:blank:]]y$", "x\ty");
  check("^x[[:blank:]]y$", "x\ny", REG_NOMATCH);

  // \w and \W: underscore matches \w; hyphen does not.
  check(R"(^x\wy$)", "x_y");
  check(R"(^x\wy$)", "x-y", REG_NOMATCH);
  check(R"(^x\Wy$)", "x-y");
  check(R"(^x\Wy$)", "x_y", REG_NOMATCH);

  // Hyphen endpoint combinations with character classes.
  check("^[--0[:alpha:]]$", "-");
  check("^[--0[:alpha:]]$", ".");
  check("^[--0[:alpha:]]$", "a");
  check("^[--0[:alpha:]]$", "1", REG_NOMATCH);

  check("^[[:alpha:]!--]$", "z");
  check("^[[:alpha:]!--]$", "!");
  check("^[[:alpha:]!--]$", "-");
  check("^[[:alpha:]!--]$", ".", REG_NOMATCH);

  // Anchored repetitions with negative rejection checks.
  check(R"(^\w*$)", "");
  check(R"(^\w*$)", "valid_word1");
  check(R"(^\w*$)", "ab!ba", REG_NOMATCH);

  check(R"(^\d*$)", "");
  check(R"(^\d*$)", "12345");
  check(R"(^\d*$)", "12a45", REG_NOMATCH);

  // Back-to-back adjacent character classes in a single bracket expression.
  check("^[[:alpha:][:blank:]]$", "a");
  check("^[[:alpha:][:blank:]]$", " ");
  check("^[[:alpha:][:blank:]]$", "\t");
  check("^[[:alpha:][:blank:]]$", "1", REG_NOMATCH);
  check("^[[:alpha:][:blank:]]$", "_", REG_NOMATCH);

  check("^[[:alpha:][:blank:]]*$", "Hello World");
  check("^[[:alpha:][:blank:]]*$", "Hello\tWorld");
  check("^[[:alpha:][:blank:]]*$", "Hello1World", REG_NOMATCH);
  check("^[[:alpha:][:blank:]]*$", "Hello_World", REG_NOMATCH);

  // Multiple classes combined with literals.
  check("^[[:digit:][:lower:]_]*$", "3abc_7");
  check("^[[:digit:][:lower:]_]*$", "3ABC_7", REG_NOMATCH);

  // Negated multi-class bracket expressions.
  check("^[^[:digit:][:space:]]*$", "abcABC_!");
  check("^[^[:digit:][:space:]]*$", "abc 123", REG_NOMATCH);
  check("^[^[:digit:][:space:]]*$", "abc1", REG_NOMATCH);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST(LlvmLibcRegexTest, NullPointerCrash) {
  regex_t preg{};
  ASSERT_DEATH([] { LIBC_NAMESPACE::regcomp(nullptr, "a", 0); },
               WITH_SIGNAL(-1));
  ASSERT_DEATH([&preg] { LIBC_NAMESPACE::regcomp(&preg, nullptr, 0); },
               WITH_SIGNAL(-1));
  ASSERT_DEATH([] { LIBC_NAMESPACE::regexec(nullptr, "a", 0, nullptr, 0); },
               WITH_SIGNAL(-1));
  ASSERT_DEATH(
      [&preg] { LIBC_NAMESPACE::regexec(&preg, nullptr, 0, nullptr, 0); },
      WITH_SIGNAL(-1));
  ASSERT_DEATH([] { LIBC_NAMESPACE::regfree(nullptr); }, WITH_SIGNAL(-1));
}
#endif // defined(LIBC_ADD_NULL_CHECKS)
