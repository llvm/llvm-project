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
#elif defined(__APPLE__)
#  include <array>
#  include <csignal>
#  include <sys/sysctl.h>
#  include <sys/types.h>
#  include <unistd.h>
#elif defined(__FreeBSD__)
#  include <array>
#  include <csignal>
#  include <libutil.h>
#  include <sys/cdefs.h>
#  include <sys/proc.h>
#  include <sys/sysctl.h>
#  include <sys/types.h>
#  include <sys/user.h>
#  include <unistd.h>
#elif defined(__linux__)
#  include <csignal>
#  include <fstream>
#  include <string>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if defined(_LIBCPP_HAS_DEBUGGING)

// breakpoint()

#  if defined(_LIBCPP_WIN32API)

void __breakpoint() noexcept { DebugBreak(); }

#  elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__linux__)

void __breakpoint() {
#    if __has_builtin(__builtin_debugtrap)
  __builtin_debugtrap();
#    else
  raise(SIGTRAP);
#    endif
}

#  else

#    error "'std::breakpoint()' is not implemented on this platform."

#  endif // defined(_LIBCPP_WIN32API)

// is_debugger_present()

#  if defined(_LIBCPP_WIN32API)

bool __is_debugger_present() noexcept { return IsDebuggerPresent(); }

#  elif defined(__APPLE__) || defined(__FreeBSD__)

// Returns true if the current process is being debugged (either
// running under the debugger or has a debugger attached post facto).
bool __is_debugger_present() noexcept {
  // Technical Q&A QA1361: Detecting the Debugger
  // https://developer.apple.com/library/archive/qa/qa1361/_index.html

  // Initialize mib, which tells 'sysctl' to fetch the information about the current process.

  array mib{CTL_KERN, KERN_PROC, KERN_PROC_PID, ::getpid()};

  // Initialize the flags so that, if 'sysctl' fails for some bizarre
  // reason, we get a predictable result.

  struct kinfo_proc info {};

  // Call sysctl.
  // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sysctl.3.html

  size_t info_size = sizeof(info);
  if (::sysctl(mib.data(), mib.size(), &info, &info_size, nullptr, 0) != 0) {
    _LIBCPP_ASSERT_INTERNAL(false, "'sysctl' runtime error");
    return false;
  }

  // If the process is being debugged if the 'P_TRACED' flag is set.
  // https://github.com/freebsd/freebsd-src/blob/7f3184ba797452703904d33377dada5f0f8eae96/sys/sys/proc.h#L822

  return ((info.kp_proc.p_flag & P_TRACED) != 0);
}

#  elif defined(__linux__)

bool __is_debugger_present() noexcept {
  // https://docs.kernel.org/filesystems/proc.html

  // Get the status information of a process by reading the file /proc/PID/status.
  // The link 'self' points to the process reading the file system.
  ifstream status_file{"/proc/self/status"};
  if (!status_file.is_open()) {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not open '/proc/self/status' for reading.");
    return false;
  }

  std::string token;
  while (status_file >> token) {
    // If the process is being debugged "TracerPid"'s value is non-zero.
    if (token == "TracerPid:") {
      int pid;
      status_file >> pid;
      return pid != 0;
    }
    getline(status_file, token);
  }

  return false;
}

#  else

#    error "'std::is_debugger_present()' is not implemented on this platform."

#  endif // defined(_LIBCPP_WIN32API)

_LIBCPP_EXPORTED_FROM_ABI void breakpoint() noexcept { __breakpoint(); }

_LIBCPP_EXPORTED_FROM_ABI void breakpoint_if_debugging() noexcept {
  if (__is_debugger_present())
    __breakpoint();
}

_LIBCPP_EXPORTED_FROM_ABI bool is_debugger_present() noexcept { return __is_debugger_present(); }

#endif // defined(_LIBCPP_HAS_DEBUGGING)

_LIBCPP_END_NAMESPACE_STD
