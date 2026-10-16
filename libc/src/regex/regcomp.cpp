//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of regcomp.
///
//===----------------------------------------------------------------------===//

#include "src/regex/regcomp.h"

#include "hdr/regex_macros.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/regex/regex_ast.h"
#include "src/__support/regex/regex_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, regcomp,
                   (regex_t *__restrict preg, const char *__restrict pattern,
                    int cflags)) {
  LIBC_CRASH_ON_NULLPTR(preg);
  LIBC_CRASH_ON_NULLPTR(pattern);

  // Note: POSIX requires callers to call regfree() before reusing a preg
  // object.  We therefore do not attempt to free any previous __internal here
  // as preg is uninitialised on first use and the pointer would be garbage.
  // However, we clear __internal and re_nsub immediately so calling regfree()
  // after a failed regcomp() is guaranteed to be a safe no-op.
  preg->__internal = nullptr;
  preg->re_nsub = 0;

  // TODO: Once cpp::unique_ptr and AllocChecker-backed cpp::make_unique are
  // available in src/__support/CPP/, manage RegexInternal via unique_ptr so
  // error paths do not require manual delete.
  AllocChecker ac;
  regex::RegexInternal *internal = new (ac) regex::RegexInternal();
  if (!ac)
    return REG_ESPACE;

  ErrorOr<regex::ExprId> cur = internal->pool.empty_str();
  if (!cur.has_value()) {
    delete internal;
    return cur.error();
  }

  // TODO: Replace direct literal concatenation with the ERE/BRE parser as
  // regular expression syntax support is wired up.
  cpp::string_view pattern_view(pattern);
  for (char c : pattern_view) {
    ErrorOr<regex::ExprId> lit = internal->pool.make_lit(c);
    if (!lit.has_value()) {
      delete internal;
      return lit.error();
    }
    cur = internal->pool.make_concat(*cur, *lit);
    if (!cur.has_value()) {
      delete internal;
      return cur.error();
    }
  }

  internal->root = *cur;
  internal->cflags = cflags;
  preg->__internal = internal;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
