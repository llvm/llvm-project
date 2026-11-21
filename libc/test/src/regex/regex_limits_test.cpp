//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Large pool-exhaustion and resource-limit tests for POSIX regex functions.
///
/// These tests intentionally drive ExprPool up to MAX_NODE_LIMIT and allocate
/// hundreds of kilobytes of heap storage.  They live in their own test binary
/// because LLVM libc's hermetic test allocator uses a fixed 1 MiB bump heap
/// where free() is a no-op, so keeping them separate preserves headroom for
/// regular functional tests in regex_basic_test.
///
//===----------------------------------------------------------------------===//

#include "hdr/regex_macros.h"
#include "hdr/types/regex_t.h"
#include "hdr/types/size_t.h"
#include "src/__support/macros/attributes.h"
#include "src/regex/regcomp.h"
#include "src/regex/regexec.h"
#include "src/regex/regfree.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcRegexLimitsTest, PoolExhaustionReturnsEspace) {
  // 9998 'a' bytes plus EmptySet (1) and EmptyStr (2) fill all 10000 ExprPool
  // slots during regcomp; regexec then hits MAX_NODE_LIMIT when interning the
  // first derivative Alt state.
  // 'static' to avoid exhausting the stack on GPU targets.
  LIBC_CONSTINIT static char pattern[10000]{};
  for (size_t i = 0; i < 9998; ++i)
    pattern[i] = 'a';
  pattern[9998] = '\0';

  regex_t preg{};
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
