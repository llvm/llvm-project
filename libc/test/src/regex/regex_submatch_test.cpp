//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Submatch and pmatch contract tests for POSIX regexec.
///
//===----------------------------------------------------------------------===//

#include "hdr/regex_macros.h"
#include "hdr/types/regex_t.h"
#include "hdr/types/regmatch_t.h"
#include "hdr/types/regoff_t.h"
#include "src/__support/CPP/scope.h"
#include "src/regex/regcomp.h"
#include "src/regex/regexec.h"
#include "src/regex/regfree.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::scope_exit;

TEST(LlvmLibcRegexSubmatchTest, PmatchInitialisedOnMatchAndPreservedOnNoMatch) {
  regex_t preg{};
  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "hello", REG_EXTENDED), 0);
  scope_exit free_guard([&preg] { LIBC_NAMESPACE::regfree(&preg); });

  regmatch_t pm[2] = {{42, 99}, {77, 88}};
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "goodbye", 2, pm, 0), REG_NOMATCH);
  EXPECT_EQ(pm[0].rm_so, regoff_t(42));
  EXPECT_EQ(pm[0].rm_eo, regoff_t(99));
  EXPECT_EQ(pm[1].rm_so, regoff_t(77));
  EXPECT_EQ(pm[1].rm_eo, regoff_t(88));

  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "say hello", 2, pm, 0), 0);
  EXPECT_EQ(pm[0].rm_so, regoff_t(-1));
  EXPECT_EQ(pm[0].rm_eo, regoff_t(-1));
  EXPECT_EQ(pm[1].rm_so, regoff_t(-1));
  EXPECT_EQ(pm[1].rm_eo, regoff_t(-1));

  // nmatch == 0 allows pmatch == nullptr even without REG_NOSUB.
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&preg, "say hello", 0, nullptr, 0), 0);

  // REG_NOSUB ignores nmatch and pmatch: nullptr succeeds and non-null pmatch
  // is left untouched on match.
  regex_t nosub{};
  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&nosub, "hello", REG_EXTENDED | REG_NOSUB),
            0);
  scope_exit free_nosub([&nosub] { LIBC_NAMESPACE::regfree(&nosub); });
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&nosub, "say hello", 2, nullptr, 0), 0);
  regmatch_t untouched[1] = {{11, 22}};
  EXPECT_EQ(LIBC_NAMESPACE::regexec(&nosub, "say hello", 1, untouched, 0), 0);
  EXPECT_EQ(untouched[0].rm_so, regoff_t(11));
  EXPECT_EQ(untouched[0].rm_eo, regoff_t(22));
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST(LlvmLibcRegexSubmatchTest, PmatchNullPointerCrash) {
  regex_t preg{};
  ASSERT_EQ(LIBC_NAMESPACE::regcomp(&preg, "a", REG_EXTENDED), 0);
  scope_exit free_guard([&preg] { LIBC_NAMESPACE::regfree(&preg); });
  ASSERT_DEATH([&preg] { LIBC_NAMESPACE::regexec(&preg, "a", 1, nullptr, 0); },
               WITH_SIGNAL(-1));
  // Null check must trigger before matching even on a non-matching input.
  ASSERT_DEATH([&preg] { LIBC_NAMESPACE::regexec(&preg, "b", 1, nullptr, 0); },
               WITH_SIGNAL(-1));
}
#endif // defined(LIBC_ADD_NULL_CHECKS)
