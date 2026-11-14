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
#include "src/__support/CPP/string_view.h"
#include "src/__support/checked_ptr.h"
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

  auto internal = make_unique<regex::RegexInternal>();
  if (!internal)
    return REG_ESPACE;

  size_t re_nsub = 0;
  ErrorOr<regex::ExprId> root = regex::parse_ere(
      cpp::string_view(pattern), cflags, internal->pool, &re_nsub);
  if (!root.has_value())
    return root.error();

  internal->root = *root;
  internal->cflags = cflags;
  preg->__internal = internal.release();
  preg->re_nsub = re_nsub;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
