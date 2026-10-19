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

  // TODO: Submatch extraction (nmatch/pmatch) and execution flags (eflags)
  // will be wired in subsequent patches.
  (void)nmatch;
  (void)pmatch;
  (void)eflags;

  // Guard against a null internal pointer (e.g. called after regfree).
  auto *internal = static_cast<regex::RegexInternal *>(preg->__internal);
  if (!internal)
    return REG_NOMATCH;

  // POSIX requires regexec to be thread-safe across concurrent calls sharing a
  // const regex_t*. Serialise access to internal->pool during lazy derivative
  // evaluation.
  cpp::lock_guard guard(internal->lock);
  switch (
      regex::match(internal->pool, internal->root, cpp::string_view(string))) {
  case regex::MatchResult::Match:
    return 0;
  case regex::MatchResult::NoMatch:
    return REG_NOMATCH;
  case regex::MatchResult::OutOfMemory:
    return REG_ESPACE;
  }
  __builtin_unreachable();
}

} // namespace LIBC_NAMESPACE_DECL
