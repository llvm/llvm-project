//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of regcomp (stub).
///
//===----------------------------------------------------------------------===//

#include "src/regex/regcomp.h"

#include "hdr/regex_macros.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, regcomp,
                   (regex_t *__restrict preg, const char *__restrict pattern,
                    int cflags)) {
  // Silencing unused parameter warning for the stub.
  (void)cflags;

  // Note: POSIX requires callers to call regfree() before reusing a preg
  // object.  We therefore do not attempt to free any previous __internal here
  // — preg is uninitialized on first use and the pointer would be garbage.

  cpp::string_view pattern_view(pattern);
  size_t len = pattern_view.size();
  AllocChecker ac;
  char *copy = new (ac) char[len + 1];
  if (!ac)
    return REG_ESPACE;

  inline_memcpy(copy, pattern, len + 1);

  // TODO: This is a stub. re_nsub is always 0 because parenthesised
  // subexpressions are not yet parsed. REG_NOSUB is effectively always active.
  preg->re_nsub = 0;
  preg->__internal = copy;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
