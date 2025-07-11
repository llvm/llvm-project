//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include "log_error_and_continue.h"

#ifdef __BIONIC__
#  include <syslog.h>
extern "C" void android_set_abort_message(const char* msg);
#endif // __BIONIC__

#if defined(__APPLE__) && __has_include(<os/reason_private.h>)
#  include <os/reason_private.h>
#   define _LIBCXXABI_USE_OS_FAULT
#endif

void __log_error_and_continue(const char* message)
{
  // On Apple platforms, use the `os_fault_with_payload` OS function that simulates a crash.
#if defined(_LIBCXXABI_USE_OS_FAULT)
  os_fault_with_payload(
      /*reason_namespace=*/OS_REASON_SECURITY,
      /*reason_code=*/0,
      /*payload=*/nullptr,
      /*payload_size=*/0,
      /*reason_string=*/message,
      /*reason_flags=*/0);

#elif defined(__BIONIC__)
  // Show error in tombstone.
  android_set_abort_message(message);

  // Show error in logcat.
  openlog("libc++", 0, 0);
  syslog(LOG_CRIT, "%s", message);
  closelog();

#else
  fprintf(stderr, "%s", message);
#endif
}
