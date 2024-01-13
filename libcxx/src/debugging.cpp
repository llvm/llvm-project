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
#  include <stdbool.h>
#  include <sys/sysctl.h>
#  include <sys/types.h>
#  include <unistd.h>
#elif defined(__linux__)
#  include <csignal>
#  include <fstream>
#  include <regex>
#  include <sstream>
#  include <string>
// Linux
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __impl {

// Linux
//   https://docs.kernel.org/filesystems/proc.html
//   https://stackoverflow.com/a/49079078
//   https://linuxsecurity.com/features/anti-debugging-for-noobs-part-1
//   https://linuxsecurity.com/features/hacker-s-corner-complete-guide-to-anti-debugging-in-linux-part-2
//   https://linuxsecurity.com/features/hacker-s-corner-complete-guide-to-anti-debugging-in-linux-part-3

// macOS
//   https://developer.apple.com/library/archive/qa/qa1361/_index.html
//   https://ladydebug.com/blog/2020/09/02/isdebuggerpresent-for-mac-osx/
//   https://github.com/freebsd/freebsd-src/blob/7f3184ba797452703904d33377dada5f0f8eae96/sys/sys/proc.h#L822

#if defined(_LIBCPP_WIN32API)

void __breakpoint() noexcept { void DebugBreak(); }

bool __is_debugger_present() noexcept { return IsDebuggerPresent(); }

#elif defined(__APPLE__) || defined(__FreeBSD__)

// TODO
void __breakpoint() {
#  ifdef _LIBCPP_HARDENING_MODE_DEBUG
#    if __has_builtin(__builtin_debugtrap)
  __builtin_debugtrap();
#    else
  raise(SIGTRAP);
#    endif
#  endif
}

bool __is_debugger_present() noexcept {
  // Technical Q&A QA1361: Detecting the Debugger
  // https://github.com/freebsd/freebsd-src/blob/7f3184ba797452703904d33377dada5f0f8eae96/sys/sys/proc.h#L822

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

  return ((info.kp_proc.p_flag & P_TRACED) != 0);
}

#elif defined(__linux__)

void __breakpoint() noexcept {
#  if defined SIGTRAP
  raise(SIGTRAP);
#  else
  raise(SIGABRT);
#  endif
}

bool __is_debugger_present() noexcept {
  // https://docs.kernel.org/filesystems/proc.html

  try {
    // Get the status information of a process by reading the file /proc/PID/status.
    // The link 'self' points to the process reading the file system.
    ifstream status_file{"/proc/self/status"};
    if (!status_file.is_open())
      return false;
#  if 0
    // string line;
    // while (std::getline(status_file, line)) {
    for (string line; std::getline(status_file, line);) {
      istringstream ss{line};
      string field;
      string value;

      ss >> field >> value;

      // TracerPid - PID of process tracing this process (0 if not, or the tracer is outside of the current pid
      // namespace).
      if ((field == "TracerPid:") && (value != "0")) {
        return true;
      }
    }
#  elif 0
    std::string line;
    while (status_file >> line) {
      if (line == "TracerPid:") {
        int pid;
        status_file >> pid;
        return pid != 0;
      }
      std::getline(status_file, line);
    }
#  else
    // This is too slow
    const regex reg_ex{R"(^TracerPid:\s+(.+)$)"};
    smatch match;
    string line;
    while (std::getline(status_file, line)) {
      if (regex_match(line, match, reg_ex)) {
        if (match[1] != "0") [[likely]]
          return true;
        return false;
      }
    }
#  endif
  } catch (...) {
    return false;
  }

  return false;
}

#else
#  define __LIBCPP_DEBUGGER_NOT_IMPLEMENTED 1
#endif

} // namespace __impl

_LIBCPP_EXPORTED_FROM_ABI void breakpoint() noexcept {
#ifdef __LIBCPP_DEBUGGER_NOT_IMPLEMENTED
  _LIBCPP_ASSERT_INTERNAL(false, "'std::is_debugger_present' is not implemented on this platform.");
#else
  __impl::__breakpoint();
#endif
}

_LIBCPP_EXPORTED_FROM_ABI void breakpoint_if_debugging() noexcept {
#ifdef __LIBCPP_DEBUGGER_NOT_IMPLEMENTED
  _LIBCPP_ASSERT_INTERNAL(false, "'std::breakpoint_if_debugging' is not implemented on this platform.");
#else
  if (__impl::__is_debugger_present())
    __impl::__breakpoint();
#endif
}

_LIBCPP_EXPORTED_FROM_ABI bool is_debugger_present() noexcept {
#ifdef __LIBCPP_DEBUGGER_NOT_IMPLEMENTED
  _LIBCPP_ASSERT_INTERNAL(false, "'std::is_debugger_present' is not implemented on this platform.");
  return false;
#else
  return __impl::__is_debugger_present();
#endif
}

#if 0
#  include <regex>
#  include <sstream>
#  include <string>

static std::string status_file_str = R"(
Name:	file:// Content
Umask:	0002
State:	R (running)
Tgid:	84655
Ngid:	0
Pid:	84655
PPid:	3287
TracerPid:	0
Uid:	1000	1000	1000	1000
Gid:	1000	1000	1000	1000
FDSize:	512
Groups:	4 24 27 30 46 122 134 135 1000 
NStgid:	84655
NSpid:	84655
NSpgid:	1923
NSsid:	1923
Kthread:	0
VmPeak:	 2387520 kB
VmSize:	 2387520 kB
VmLck:	       0 kB
VmPin:	       0 kB
VmHWM:	   71680 kB
VmRSS:	   71680 kB
RssAnon:	   11904 kB
RssFile:	   58752 kB
RssShmem:	    1024 kB
VmData:	   30796 kB
VmStk:	     148 kB
VmExe:	     700 kB
VmLib:	  115052 kB
VmPTE:	     420 kB
VmSwap:	       0 kB
HugetlbPages:	       0 kB
CoreDumping:	0
THP_enabled:	1
untag_mask:	0xffffffffffffffff
Threads:	21
SigQ:	0/30009
SigPnd:	0000000000000000
ShdPnd:	0000000000000000
SigBlk:	0000000000000000
SigIgn:	0000000000011002
SigCgt:	0000000f408004f8
CapInh:	0000000000000000
CapPrm:	0000000000000000
CapEff:	0000000000000000
CapBnd:	000001ffffffffff
CapAmb:	0000000000000000
NoNewPrivs:	1
Seccomp:	2
Seccomp_filters:	3
Speculation_Store_Bypass:	thread vulnerable
SpeculationIndirectBranch:	conditional enabled
Cpus_allowed:	ff
Cpus_allowed_list:	0-7
Mems_allowed:	00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000001
Mems_allowed_list:	0
voluntary_ctxt_switches:	31
nonvoluntary_ctxt_switches:	18
)";

static void DebuggerPresent01(benchmark::State& state) {
  // Code before the loop is not measured
  std::istringstream status_file{status_file_str};
  
  for (auto _ : state) {
    for (std::string line; std::getline(status_file, line);) {
      std::istringstream ss{line};
      std::string field;
      std::string value;
      ss >> field >> value;
      if ((field == "TracerPid:") && (value != "0")) {
        goto DP01;
      }
    }
  }
DP01:
}
BENCHMARK(DebuggerPresent01);

static void DebuggerPresent02(benchmark::State& state) {
  // Code before the loop is not measured
  std::istringstream status_file{status_file_str};
  
  for (auto _ : state) {
    std::string line;
    while (status_file >> line) {
      if (line == "TracerPid:") {
        int pid;
        status_file >> pid;
        auto a = pid != 0;
        benchmark::DoNotOptimize(a);
        goto DP02;
      }
      std::getline(status_file, line);
    }
DP02:
  }
}
BENCHMARK(DebuggerPresent02);

static void DebuggerPresent03(benchmark::State& state) {
  // Code before the loop is not measured
  std::istringstream status_file{status_file_str};
  
  for (auto _ : state) {
    std::smatch match;
    std::string line;
    const std::regex reg_ex{R"(^TracerPid:\s+(.+)$)"};
    while (std::getline(status_file, line)) {
      if (std::regex_match(line, match, reg_ex)) {
        if (match[1] != "0") [[likely]]
          goto DP03;
        goto DP03;
      }
    }
DP03:
  }
}
BENCHMARK(DebuggerPresent03);

#endif

_LIBCPP_END_NAMESPACE_STD
