//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of verr.
///
//===----------------------------------------------------------------------===//

#include "src/err/verr.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/arg_list.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/err/report.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

[[noreturn]] LLVM_LIBC_FUNCTION(void, verr,
                                (int eval, const char *fmt, va_list args)) {
  int saved_errno = libc_errno;
  internal::ArgList arg_list(args);
  err_reporting::report(true, saved_errno, fmt, arg_list);
  internal::exit(eval);
}

} // namespace LIBC_NAMESPACE_DECL
