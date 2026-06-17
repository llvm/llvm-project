//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of regexec (stub).
///
//===----------------------------------------------------------------------===//

#include "src/regex/regexec.h"

#include "hdr/regex_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_strstr.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, regexec,
                   (const regex_t *__restrict preg,
                    const char *__restrict string, size_t nmatch,
                    regmatch_t *__restrict pmatch, int eflags)) {
  // TODO: This is a stub. The following are not yet implemented:
  //   - Regex metacharacters (., *, +, ?, [], {}, (), |, ^, $).
  //   - REG_EXTENDED / REG_ICASE / REG_NEWLINE compile flags.
  //   - REG_NOTBOL / REG_NOTEOL eflags.
  //   - pmatch[] filling (subexpression offsets).
  //   - Only literal substring search (strstr) is performed.
  (void)nmatch;
  (void)pmatch;
  (void)eflags;

  // Guard against a null internal pointer (e.g. called after regfree).
  const char *pattern = static_cast<const char *>(preg->__internal);
  if (!pattern)
    return REG_NOMATCH;

  // An empty pattern always matches.
  if (*pattern == '\0')
    return 0;

  // Use inline_strstr for literal substring matching.
  auto comp = [](char l, char r) -> int { return l - r; };
  if (inline_strstr(string, pattern, comp))
    return 0;

  return REG_NOMATCH;
}

} // namespace LIBC_NAMESPACE_DECL
