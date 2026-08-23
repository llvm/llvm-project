//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/overridable_function.h"

#include <__assert>
#include <__config>
#include <debugging>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#if defined(_AIX)
#  include <charconv>
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
#  include <sys/sysctl.h>
#  include <sys/types.h>
#  include <unistd.h>
#elif defined(__linux__)
#  include <array>
#  include <cstring>
#  include <fcntl.h>
#  include <span>
#  include <unistd.h>
#endif

_LIBCPP_EXPORTED_FROM_ABI volatile int __gnu_cxx::debugger_signal_for_breakpoint = 0;

_LIBCPP_DIAGNOSTIC_PUSH
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wmissing-prototypes")

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

// `breakpoint()` implementation

void __breakpoint() noexcept {
#if defined(_WIN32)
  DebugBreak();
#else
  raise(SIGTRAP);
#endif // defined(_WIN32)
}

// `is_debugger_present()` implementation

OVERRIDABLE_FUNCTION bool is_debugger_present() noexcept {
  if (__gnu_cxx::debugger_signal_for_breakpoint != 0)
    return true;

#if defined(_WIN32)

  return IsDebuggerPresent();

#elif defined(_AIX)

  // Get the status information of a process by memory mapping the file /proc/PID/status.
  // https://www.ibm.com/docs/en/aix/7.3?topic=files-proc-file
  char __filename[] = "/proc/4294967295/status";
  if (auto [__ptr, __ec] = std::to_chars(__filename + 6, __filename + 16, ::getpid()); __ec == std::errc()) {
    ::strcpy(__ptr, "/status");
  } else {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not convert pid to cstring.");
    return false;
  }

  int __fd = ::open(__filename, O_RDONLY);
  if (__fd < 0) {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not open '/proc/{pid}/status' for reading.");
    return false;
  }

  pstatus_t __status;
  if (::read(__fd, &__status, sizeof(pstatus_t)) < static_cast<ssize_t>(sizeof(pstatus_t))) {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not read from '/proc/{pid}/status'.");
    return false;
  }

  if (__status.pr_flag & STRC)
    return true;

  return false;

#elif defined(__APPLE__) || defined(__FreeBSD__)

  // Returns true if the current process is being debugged (either
  // running under the debugger or has a debugger attached post facto).

  // Technical Q&A QA1361: Detecting the Debugger
  // https://developer.apple.com/library/archive/qa/qa1361/_index.html

  // Initialize mib, which tells 'sysctl' to fetch the information about the current process.

  array __mib{CTL_KERN, KERN_PROC, KERN_PROC_PID, ::getpid()};

  // Initialize the flags so that, if 'sysctl' fails for some bizarre
  // reason, we get a predictable result.

  struct kinfo_proc __info{};

  // Call sysctl.
  // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sysctl.3.html

  size_t __info_size = sizeof(__info);
  if (::sysctl(__mib.data(), __mib.size(), &__info, &__info_size, nullptr, 0) != 0) {
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

#elif defined(__linux__)

  // https://docs.kernel.org/filesystems/proc.html
  alignas(8) array<char, 256 + 1> __buffer;
  constexpr auto __tracer_key_ = span("\nTracerPid:\t");

  auto __result = [&__buffer] {
    int __buf_read      = ::open("/proc/self/status", O_RDONLY | O_CLOEXEC);
    const auto __result = ::read(__buf_read, __buffer.data(), __buffer.size() - 1);
    ::close(__buf_read);
    return __result;
  }();

  if (__result < 80) {
    return false;
  }

  __buffer[__result] = '\0';

  char* __pos = std::strstr(__buffer.data() + 64, __tracer_key_.data());
  return __pos != nullptr && __pos[__tracer_key_.size() - 1] != '0';

#else

  // The implementation returns 'false' by default.
  return false;

#endif // defined(_WIN32)
}

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_DIAGNOSTIC_POP
