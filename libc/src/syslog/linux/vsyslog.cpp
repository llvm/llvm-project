//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of vsyslog.
///
//===----------------------------------------------------------------------===//

#include "src/syslog/vsyslog.h"
#include "src/__support/arg_list.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/syslog/linux/syslog_internal.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, vsyslog,
                   (int priority, const char *format, va_list args)) {
  LIBC_CRASH_ON_NULLPTR(format);
  internal::ArgList arg_list(args);
  syslog_internal::log_message(priority, format, arg_list);
}

} // namespace LIBC_NAMESPACE_DECL
