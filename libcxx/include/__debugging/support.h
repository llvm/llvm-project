#ifndef _LIBCPP___DEBUGGING_SUPPORT_H
#define _LIBCPP___DEBUGGING_SUPPORT_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if defined(_WIN32)
#  include <__debugging/support/windows.h>
#elif defined(_AIX)
#  include <__debugging/support/aix.h>
#elif defined(__APPLE__) || defined(__FreeBSD__)
#  include <__debugging/support/bsd_like.h>
#elif defined(__linux__)
#  include <__debugging/support/linux.h>
#else

_LIBCPP_BEGIN_NAMESPACE_STD

#  if _LIBCPP_STD_VER >= 26

_LIBCPP_HIDE_FROM_ABI inline bool __libcpp_is_debugger_present() noexcept { return false; }

#  endif

_LIBCPP_END_NAMESPACE_STD

#endif // defined(_WIN32)

#endif // _LIBCPP___DEBUGGING_SUPPORT_H
