//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__assert>
#include <__config>
#include <debugging>

#if defined(_LIBCPP_WIN32API)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#elif defined(_AIX)
#  include <charconv>
#  include <csignal>
#  include <cstring>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/proc.h>
#  include <sys/procfs.h>
#  include <sys/types.h>
#  include <unistd.h>
#elif defined(__APPLE__) || defined(__FreeBSD__)
#  if defined(__FreeBSD__) // Include order matters.
#    include <libutil.h>
#    include <sys/param.h>
#    include <sys/proc.h>
#    include <sys/user.h>
#  endif // defined(__FreeBSD__)
#  include <array>
#  include <csignal>
#  include <sys/sysctl.h>
#  include <sys/types.h>
#  include <unistd.h>
#elif defined(__PICOLIBC__)
#  include <csignal>
#elif defined(__linux__)
#  include <csignal>
#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// `breakpoint()` implementation

_LIBCPP_EXPORTED_FROM_ABI void __breakpoint() noexcept {
#if defined(_LIBCPP_WIN32API)
  DebugBreak();
#elif defined(_AIX) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__PICOLIBC__) || defined(__linux__)
  raise(SIGTRAP);
#else
#  error "'std::breakpoint()' is not implemented on this platform."
#endif // defined(_LIBCPP_WIN32API)
}

// `is_debugger_present()` implementation

#if defined(_LIBCPP_WIN32API)

static bool __is_debugger_present() noexcept { return IsDebuggerPresent(); }

#elif defined(_AIX)

static bool __is_debugger_present() noexcept {
  // Get the status information of a process by memory mapping the file /proc/PID/status.
  // https://www.ibm.com/docs/en/aix/7.3?topic=files-proc-file
  char __filename[] = "/proc/4294967295/status";
  if (auto [__ptr, __ec] = to_chars(__filename + 6, __filename + 16, getpid()); __ec == errc()) {
    strcpy(__ptr, "/status");
  } else {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not convert pid to cstring.");
    return false;
  }

  int __fd = open(__filename, O_RDONLY);
  if (__fd < 0) {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not open '/proc/{pid}/status' for reading.");
    return false;
  }

  pstatus_t __status;
  if (read(__fd, &__status, sizeof(pstatus_t)) < static_cast<ssize_t>(sizeof(pstatus_t))) {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not read from '/proc/{pid}/status'.");
    return false;
  }

  if (__status.pr_flag & STRC)
    return true;

  return false;
}

#elif defined(__APPLE__) || defined(__FreeBSD__)

// Returns true if the current process is being debugged (either
// running under the debugger or has a debugger attached post facto).
static bool __is_debugger_present() noexcept {
  // Technical Q&A QA1361: Detecting the Debugger
  // https://developer.apple.com/library/archive/qa/qa1361/_index.html

  // Initialize mib, which tells 'sysctl' to fetch the information about the current process.

  array __mib{CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()};

  // Initialize the flags so that, if 'sysctl' fails for some bizarre
  // reason, we get a predictable result.

  struct kinfo_proc __info {};

  // Call sysctl.
  // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sysctl.3.html

  size_t __info_size = sizeof(__info);
  if (sysctl(__mib.data(), __mib.size(), &__info, &__info_size, nullptr, 0) != 0) {
    _LIBCPP_ASSERT_INTERNAL(false, "'sysctl' runtime error");
    return false;
  }

  // If the process is being debugged if the 'P_TRACED' flag is set.
  // https://github.com/freebsd/freebsd-src/blob/7f3184ba797452703904d33377dada5f0f8eae96/sys/sys/proc.h#L822

#  if defined(__FreeBSD__)
  const auto __p_flag = __info.ki_flag;
#  else // __APPLE__
  const auto __p_flag = __info.kp_proc.p_flag;
#  endif

  return ((__p_flag & P_TRACED) != 0);
}

#elif defined(__linux__)

static bool __is_debugger_present() noexcept {
#  if defined(_LIBCPP_HAS_NO_FILESYSTEM)
  _LIBCPP_ASSERT_INTERNAL(false,
                          "Function is not available. Could not open '/proc/self/status' for reading, libc++ was "
                          "compiled with _LIBCPP_HAS_NO_FILESYSTEM.");
  return false;
#  else
  // https://docs.kernel.org/filesystems/proc.html

  // Get the status information of a process by reading the file /proc/PID/status.
  // The link 'self' points to the process reading the file system.
  FILE* __proc_status_fp = fopen("/proc/self/status", "r");
  if (__proc_status_fp == nullptr) {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not open '/proc/self/status' for reading.");
    return false;
  }

  char* __line               = nullptr;
  size_t __lineLen           = 0;
  const char* __tokenStr     = "TracerPid:";
  bool __is_debugger_present = false;

  while ((getline(&__line, &__lineLen, __proc_status_fp)) != -1) {
    // If the process is being debugged "TracerPid"'s value is non-zero.
    char* __tokenPos = strstr(__line, __tokenStr);
    if (__tokenPos == nullptr) {
      continue;
    }

    __is_debugger_present = (atoi(__tokenPos + strlen(__tokenStr)) != 0);
    break;
  }

  free(__line);
  fclose(__proc_status_fp);

  return __is_debugger_present;
#  endif // _LIBCPP_HAS_NO_FILESYSTEM
}

#else

static bool __is_debugger_present() noexcept { return false; }

#endif // defined(_LIBCPP_WIN32API)

_LIBCPP_EXPORTED_FROM_ABI _LIBCPP_WEAK bool is_debugger_present() noexcept { return __is_debugger_present(); }

_LIBCPP_END_NAMESPACE_STD
