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

TEST(LlvmLibcRegexTest, BasicLiteralRoundTrip) {
  regex_t preg;
  ASSERT_EQ(0,
            LIBC_NAMESPACE::regcomp(&preg, "hello", REG_EXTENDED | REG_NOSUB));
  ASSERT_EQ(0,
            LIBC_NAMESPACE::regexec(&preg, "say hello world", 0, nullptr, 0));
  ASSERT_EQ(REG_NOMATCH,
            LIBC_NAMESPACE::regexec(&preg, "goodbye", 0, nullptr, 0));
  LIBC_NAMESPACE::regfree(&preg);
}

TEST(LlvmLibcRegexTest, MismatchCases) {
  regex_t preg;
  // Partial match
  ASSERT_EQ(0,
            LIBC_NAMESPACE::regcomp(&preg, "hello", REG_EXTENDED | REG_NOSUB));
  ASSERT_EQ(REG_NOMATCH, LIBC_NAMESPACE::regexec(&preg, "hell", 0, nullptr, 0));
  LIBC_NAMESPACE::regfree(&preg);

  // Case sensitivity
  ASSERT_EQ(0,
            LIBC_NAMESPACE::regcomp(&preg, "Hello", REG_EXTENDED | REG_NOSUB));
  ASSERT_EQ(REG_NOMATCH,
            LIBC_NAMESPACE::regexec(&preg, "hello", 0, nullptr, 0));
  LIBC_NAMESPACE::regfree(&preg);

  // Empty string vs non-empty pattern
  ASSERT_EQ(0, LIBC_NAMESPACE::regcomp(&preg, "a", REG_EXTENDED | REG_NOSUB));
  ASSERT_EQ(REG_NOMATCH, LIBC_NAMESPACE::regexec(&preg, "", 0, nullptr, 0));
  LIBC_NAMESPACE::regfree(&preg);
}

TEST(LlvmLibcRegexTest, EmptyString) {
  regex_t preg;
  ASSERT_EQ(0, LIBC_NAMESPACE::regcomp(&preg, "", REG_EXTENDED | REG_NOSUB));
  ASSERT_EQ(0, LIBC_NAMESPACE::regexec(&preg, "anything", 0, nullptr, 0));
  ASSERT_EQ(0, LIBC_NAMESPACE::regexec(&preg, "", 0, nullptr, 0));
  LIBC_NAMESPACE::regfree(&preg);
}

TEST(LlvmLibcRegexTest, ExactMatch) {
  regex_t preg;
  ASSERT_EQ(0,
            LIBC_NAMESPACE::regcomp(&preg, "test", REG_EXTENDED | REG_NOSUB));
  ASSERT_EQ(0, LIBC_NAMESPACE::regexec(&preg, "test", 0, nullptr, 0));
  LIBC_NAMESPACE::regfree(&preg);
}

TEST(LlvmLibcRegexTest, NullByteStopsParsing) {
  regex_t preg;
  ASSERT_EQ(0,
            LIBC_NAMESPACE::regcomp(&preg, "match", REG_EXTENDED | REG_NOSUB));
  ASSERT_EQ(REG_NOMATCH,
            LIBC_NAMESPACE::regexec(&preg, "doesn't \0 match", 0, nullptr, 0));
  LIBC_NAMESPACE::regfree(&preg);
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

  // One additional repeated byte (9999 'a's) hits MAX_NODE_LIMIT in
  // make_concat; a novel byte ('b' at index 9998) hits MAX_NODE_LIMIT in
  // make_lit. Calling regfree afterwards must be a safe no-op in both cases.
  pattern[9998] = 'a';
  pattern[9999] = '\0';
  EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, pattern, REG_EXTENDED | REG_NOSUB),
            REG_ESPACE);
  LIBC_NAMESPACE::regfree(&preg);

  pattern[9998] = 'b';
  EXPECT_EQ(LIBC_NAMESPACE::regcomp(&preg, pattern, REG_EXTENDED | REG_NOSUB),
            REG_ESPACE);
  LIBC_NAMESPACE::regfree(&preg);
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
