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
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/regex/regex_ast.h"
#include "src/__support/regex/regex_internal.h"
#include "src/__support/regex/regex_parser.h"

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

  // TODO: Enable BRE (!(cflags & REG_EXTENDED)), REG_ICASE, and REG_NOSPEC as
  // each feature is implemented in subsequent patches.
  if (!(cflags & REG_EXTENDED) ||
      (cflags & ~(REG_EXTENDED | REG_NOSUB | REG_NEWLINE)) != 0)
    return REG_BADPAT;

  // TODO: Once cpp::unique_ptr and AllocChecker-backed cpp::make_unique are
  // available in src/__support/CPP/, manage RegexInternal via unique_ptr so
  // error paths do not require a scope guard.
  AllocChecker ac;
  regex::RegexInternal *internal = new (ac) regex::RegexInternal();
  if (!ac)
    return REG_ESPACE;
  cpp::scope_exit internal_guard([internal] { delete internal; });

  ErrorOr<regex::ExprId> root =
      regex::parse_ere(cpp::string_view(pattern), cflags, internal->pool);
  if (!root.has_value())
    return root.error();

  internal_guard.release();
  internal->root = *root;
  internal->cflags = cflags;
  preg->__internal = internal;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
