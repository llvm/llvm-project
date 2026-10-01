//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of errx.
///
//===----------------------------------------------------------------------===//

#include "src/err/errx.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/arg_list.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/err/report.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

[[noreturn]] LLVM_LIBC_FUNCTION(void, errx, (int eval, const char *fmt, ...)) {
  int saved_errno = libc_errno;
  va_list args;
  va_start(args, fmt);
  internal::ArgList arg_list(args);
  err_reporting::report(false, saved_errno, fmt, arg_list);
  va_end(args);
  internal::exit(eval);
}

} // namespace LIBC_NAMESPACE_DECL
