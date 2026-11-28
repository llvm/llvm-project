//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of regexec.
///
//===----------------------------------------------------------------------===//

#include "src/regex/regexec.h"

#include "hdr/regex_macros.h"
#include "src/__support/CPP/mutex.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/regex/regex_internal.h"
#include "src/__support/regex/regex_matcher.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, regexec,
                   (const regex_t *__restrict preg,
                    const char *__restrict string, size_t nmatch,
                    regmatch_t *__restrict pmatch, int eflags)) {
  LIBC_CRASH_ON_NULLPTR(preg);
  LIBC_CRASH_ON_NULLPTR(string);

  if ((eflags & ~(REG_NOTBOL | REG_NOTEOL)) != 0)
    return REG_BADPAT;

  // Guard against a null internal pointer (e.g. called after regfree).
  auto *internal = static_cast<regex::RegexInternal *>(preg->__internal);
  if (!internal)
    return REG_NOMATCH;

  const bool want_pmatch = nmatch > 0 && !(internal->cflags & REG_NOSUB);
  if (want_pmatch)
    LIBC_CRASH_ON_NULLPTR(pmatch);

  // POSIX requires regexec to be thread-safe across concurrent calls sharing a
  // const regex_t*. Serialise access to internal->pool during lazy derivative
  // evaluation, then release the lock before writing caller-owned pmatch.
  regex::MatchResult res = regex::MatchResult::NoMatch;
  {
    cpp::lock_guard guard(internal->lock);
    res = regex::match(*internal, cpp::string_view(string), eflags);
  }
  switch (res) {
  case regex::MatchResult::Match:
    // TODO: Populate leftmost-longest match offsets in subsequent patches.
    // Initialise pmatch to {-1, -1} so callers never observe uninitialised
    // stack memory.
    if (want_pmatch) {
      for (size_t i = 0; i < nmatch; ++i) {
        pmatch[i].rm_so = -1;
        pmatch[i].rm_eo = -1;
      }
    }
    return 0;
  case regex::MatchResult::NoMatch:
    return REG_NOMATCH;
  case regex::MatchResult::OutOfMemory:
    return REG_ESPACE;
  }
  LIBC_ASSERT(false);
  __builtin_unreachable();
}

} // namespace LIBC_NAMESPACE_DECL
