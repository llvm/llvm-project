//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of warnx.
///
//===----------------------------------------------------------------------===//

#include "src/err/warnx.h"
#include "src/__support/arg_list.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/err/report.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, warnx, (const char *fmt, ...)) {
  LIBC_CRASH_ON_NULLPTR(fmt);
  va_list args;
  va_start(args, fmt);
  internal::ArgList arg_list(args);
  err_reporting::report(false, fmt, arg_list);
  va_end(args);
}

} // namespace LIBC_NAMESPACE_DECL
