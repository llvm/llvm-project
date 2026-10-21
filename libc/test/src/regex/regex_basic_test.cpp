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
  const char *const bad_patterns[] = {"a*",    "a+",    "a?",    "(a)",
                                      "[a]",   "a{1}",  "a|b",   R"(\1)",
                                      R"(\b)", R"(\w)", R"(\s)", R"(\d)"};
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
