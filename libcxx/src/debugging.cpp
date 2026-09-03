//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__assert>
#include <__config>
#include <csignal>
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

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

void __breakpoint() noexcept {
#if defined(_WIN32)
  DebugBreak();
#else
  raise(SIGTRAP);
#endif // defined(_WIN32)
}

[[__gnu__::__weak__]] bool is_debugger_present() noexcept {
#if defined(_WIN32)

  return IsDebuggerPresent();

#elif defined(_AIX)

  // Get the status information of a process by memory mapping the file /proc/PID/status.
  // https://www.ibm.com/docs/en/aix/7.3?topic=files-proc-file
  char filename[] = "/proc/4294967295/status";
  if (auto [ptr, ec] = std::to_chars(filename + 6, filename + 16, ::getpid()); ec == std::errc()) {
    ::strcpy(ptr, "/status");
  } else {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not convert pid to cstring.");
    return false;
  }

  int fd = ::open(filename, O_RDONLY);
  if (fd < 0) {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not open '/proc/{pid}/status' for reading.");
    return false;
  }

  ::pstatus_t status;
  if (::read(fd, &status, sizeof(::pstatus_t)) < static_cast<ssize_t>(sizeof(::pstatus_t))) {
    _LIBCPP_ASSERT_INTERNAL(false, "Could not read from '/proc/{pid}/status'.");
    return false;
  }

  if (status.pr_flag & STRC)
    return true;

  return false;

#elif defined(__APPLE__) || defined(__FreeBSD__)
  // Technical Q&A QA1361: Detecting the Debugger
  // https://developer.apple.com/library/archive/qa/qa1361/_index.html

  // Initialize mib, which tells 'sysctl' to fetch the information about the current process.

  array<int, 4> mib{CTL_KERN, KERN_PROC, KERN_PROC_PID, ::getpid()};

  // Initialize the flags so that, if 'sysctl' fails for some bizarre
  // reason, we get a predictable result.

  ::kinfo_proc info{};

  // Call sysctl.
  // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sysctl.3.html

  size_t info_size = sizeof(info);
  if (::sysctl(mib.data(), mib.size(), &info, &info_size, nullptr, 0) != 0) {
    _LIBCPP_ASSERT_INTERNAL(false, "'sysctl' runtime error");
    return false;
  }

  // The process is being debugged if the 'P_TRACED' flag is set.
  // https://github.com/freebsd/freebsd-src/blob/7f3184ba797452703904d33377dada5f0f8eae96/sys/sys/proc.h#L822

#  if defined(__FreeBSD__)
  const auto p_flag = info.ki_flag;
#  else // __APPLE__
  const auto p_flag = info.kp_proc.p_flag;
#  endif

  return ((p_flag & P_TRACED) != 0);

#elif defined(__linux__)

  // https://docs.kernel.org/filesystems/proc.html
  alignas(8) array<char, 256 + 1> buffer;
  constexpr std::span<const char> tracer_key("\nTracerPid:\t");

  int buf_read      = ::open("/proc/self/status", O_RDONLY | O_CLOEXEC);
  const auto result = ::read(buf_read, buffer.data(), buffer.size() - 1);
  ::close(buf_read);

  if (result < 80) {
    return false;
  }

  buffer[result] = '\0';

  char* pos = std::strstr(buffer.data() + 64, tracer_key.data());
  return pos != nullptr && pos[tracer_key.size() - 1] != '0';

#else

  return false;

#endif // defined(_WIN32)
}

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS

_LIBCPP_END_NAMESPACE_STD
