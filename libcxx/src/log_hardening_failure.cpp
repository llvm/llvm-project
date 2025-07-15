//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__log_hardening_failure>
#include <cstdio>

#ifdef __BIONIC__
#  include <syslog.h>
extern "C" void android_set_abort_message(const char* msg);
#endif // __BIONIC__

#if defined(__APPLE__) && __has_include(<os/reason_private.h>)
#  include <TargetConditionals.h>
#  include <os/reason_private.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

void __log_hardening_failure(const char* message) noexcept {
  // Always log the message to `stderr` in case the platform-specific system calls fail.
  fputs(message, stderr);

  // On Apple platforms, use the `os_fault_with_payload` OS function that simulates a crash.
#if defined(__APPLE__) && __has_include(<os/reason_private.h>) && !TARGET_OS_SIMULATOR
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
#endif
}

_LIBCPP_END_NAMESPACE_STD
