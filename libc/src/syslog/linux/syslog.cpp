//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of syslog.
///
//===----------------------------------------------------------------------===//

#include "src/syslog/syslog.h"
#include "src/__support/arg_list.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/syslog/linux/syslog_internal.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, syslog, (int priority, const char *format, ...)) {
  LIBC_CRASH_ON_NULLPTR(format);
  va_list vlist;
  va_start(vlist, format);
  internal::ArgList args(vlist);
  va_end(vlist);
  syslog_internal::log_message(priority, format, args);
}

} // namespace LIBC_NAMESPACE_DECL
