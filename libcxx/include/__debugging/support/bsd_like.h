#ifndef _LIBCPP___DEBUGGING_SUPPORT_BSD_LIKE_H
#define _LIBCPP___DEBUGGING_SUPPORT_BSD_LIKE_H

#include <__config>

#if defined(__FreeBSD__) // Include order matters.
#  include <libutil.h>
#  include <sys/param.h>
#  include <sys/proc.h>
#  include <sys/user.h>
#endif // defined(__FreeBSD__)
#include <array>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

_LIBCPP_HIDE_FROM_ABI inline bool __libcpp_is_debugger_present() noexcept {
  // Technical Q&A QA1361: Detecting the Debugger
  // https://developer.apple.com/library/archive/qa/qa1361/_index.html

  // Initialize mib, which tells 'sysctl' to fetch the information about the current process.

  array<int, 4> __mib{CTL_KERN, KERN_PROC, KERN_PROC_PID, ::getpid()};

  // Initialize the flags so that, if 'sysctl' fails for some bizarre
  // reason, we get a predictable result.

  ::kinfo_proc __info{};

  // Call sysctl.
  // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/sysctl.3.html

  size_t __info_size = sizeof(__info);
  if (::sysctl(__mib.data(), __mib.size(), &__info, &__info_size, nullptr, 0) != 0) {
    _LIBCPP_ASSERT_INTERNAL(false, "'sysctl' runtime error");
    return false;
  }

  // The process is being debugged if the 'P_TRACED' flag is set.
  // https://github.com/freebsd/freebsd-src/blob/7f3184ba797452703904d33377dada5f0f8eae96/sys/sys/proc.h#L822

#  if defined(__FreeBSD__)
  const auto __p_flag = __info.ki_flag;
#  else // __APPLE__
  const auto __p_flag = __info.kp_proc.p_flag;
#  endif

  return ((__p_flag & P_TRACED) != 0);
}

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___DEBUGGING_SUPPORT_BSD_LIKE_H
