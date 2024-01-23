//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <debugging>

#if defined(_LIBCPP_WIN32API)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#elif defined(__APPLE__) || defined(__FreeBSD__)
#  include <csignal>
#  include <sys/sysctl.h>
#  include <sys/types.h>
#  include <unistd.h>
#elif defined(__linux__)
#  include <csignal>
#  include <fstream>
#  include <sstream>
#  include <string>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// breakpoint()

#if defined(_LIBCPP_WIN32API)

void __breakpoint() noexcept { void DebugBreak(); }

#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__linux__)

void __breakpoint() {
#  if __has_builtin(__builtin_debugtrap)
  __builtin_debugtrap();
#  else
  raise(SIGTRAP);
#  endif
}

#else

void __breakpoint() noexcept {
  static_assert(false, "'std::breakpoint()' is not implemented on this platform.");
  return false;
}

#endif // defined(_LIBCPP_WIN32API)

// is_debugger_present()

#if defined(_LIBC_WIN32API)

bool __is_debugger_present() noexcept { return IsDebuggerPresent(); }

#elif defined(__APPLE__) || defined(__FreeBSD__)

bool __is_debugger_present() noexcept {
  // Technical Q&A QA1361: Detecting the Debugger
  // https://developer.apple.com/library/archive/qa/qa1361/_index.html

  int junk;
  int mib[4];
  struct kinfo_proc info;
  size_t size;

  // Initialize the flags so that, if sysctl fails for some bizarre
  // reason, we get a predictable result.

  info.kp_proc.p_flag = 0;

  // Initialize mib, which tells sysctl the info we want, in this case
  // we're looking for information about a specific process ID.

  mib[0] = CTL_KERN;
  mib[1] = KERN_PROC;
  mib[2] = KERN_PROC_PID;
  mib[3] = getpid();

  // Call sysctl.

  size = sizeof(info);
  junk = sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &size, NULL, 0);
  _LIBCPP_ASSERT_INTERNAL(junk == 0, "'sysctl' runtime error");

  // We're being debugged if the P_TRACED flag is set.
  // https://github.com/freebsd/freebsd-src/blob/7f3184ba797452703904d33377dada5f0f8eae96/sys/sys/proc.h#L822

  return ((info.kp_proc.p_flag & P_TRACED) != 0);
}

#elif defined(__linux__)

bool __is_debugger_present() noexcept {
  // https://docs.kernel.org/filesystems/proc.html

  try {
    // Get the status information of a process by reading the file /proc/PID/status.
    // The link 'self' points to the process reading the file system.
    ifstream status_file{"/proc/self/status"};
    if (!status_file.is_open()) {
      _LIBCPP_ASSERT_INTERNAL(false, "Could not open '/proc/self/status' for reading.");
      return false;
    }

    const string tracerPid{"TracerPid"};
    for (string line; getline(status_file, line);) {
      if (line.starts_with(tracerPid)) {
        string value = line.substr(tracerPid.size() + 1);
        return stoll(value) != 0;
      }
    }
  } catch (...) {
    _LIBCPP_ASSERT_INTERNAL(false, "Failed to read '/proc/self/status'.");
    return false;
  }

  return false;
}

#else

bool __is_debugger_present() noexcept {
  static_assert(false, "'std::is_debugger_present()' is not implemented on this platform.");
  return false;
}

#endif // defined(_LIBCPP_WIN32API)

_LIBCPP_EXPORTED_FROM_ABI void breakpoint() noexcept { __breakpoint(); }

_LIBCPP_EXPORTED_FROM_ABI void breakpoint_if_debugging() noexcept {
  if (__is_debugger_present())
    __breakpoint();
}

_LIBCPP_EXPORTED_FROM_ABI bool is_debugger_present() noexcept { return __is_debugger_present(); }

_LIBCPP_END_NAMESPACE_STD
