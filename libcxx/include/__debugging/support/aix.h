// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___DEBUGGING_SUPPORT_AIX_H
#define _LIBCPP___DEBUGGING_SUPPORT_AIX_H

#include <__config>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/proc.h>
#include <sys/procfs.h>
#include <sys/types.h>
#include <unistd.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

_LIBCPP_HIDE_FROM_ABI inline bool __libcpp_is_debugger_present() noexcept {
  // Get the status information of a process by memory mapping the file /proc/PID/status.
  // https://www.ibm.com/docs/en/aix/7.3?topic=files-proc-file
  char __filename[] = "/proc/4294967295/status";
  if (auto [ptr, ec] = std::to_chars(__filename + 6, __filename + 16, ::getpid()); ec == std::errc()) {
    ::strcpy(ptr, "/status");
  } else {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not convert pid to cstring.");
    return false;
  }

  int __fd = ::open(__filename, O_RDONLY);
  if (__fd < 0) {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not open '/proc/{pid}/status' for reading.");
    return false;
  }

  ::pstatus_t __status;
  if (::read(__fd, &__status, sizeof(::pstatus_t)) < static_cast<ssize_t>(sizeof(::pstatus_t))) {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not read from '/proc/{pid}/status'.");
    return false;
  }

  if (__status.pr_flag & STRC)
    return true;

  return false;
}

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___DEBUGGING_SUPPORT_AIX_H
