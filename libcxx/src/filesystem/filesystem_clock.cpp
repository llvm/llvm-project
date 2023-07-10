//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <chrono>
#include <filesystem>
#include <time.h>

#include "time_utils.h"

#if defined(_LIBCPP_WIN32API)
# define WIN32_LEAN_AND_MEAN
# define NOMINMAX
# include <windows.h>
#endif

#if !defined(CLOCK_REALTIME) && !defined(_LIBCPP_WIN32API)
# include <sys/time.h> // for gettimeofday and timeval
#endif

_LIBCPP_BEGIN_NAMESPACE_FILESYSTEM

const bool _FilesystemClock::is_steady;

_FilesystemClock::time_point _FilesystemClock::now() noexcept {
  typedef chrono::duration<rep> __secs;
#if defined(_LIBCPP_WIN32API)
  typedef chrono::duration<rep, nano> __nsecs;
  FILETIME time;
  GetSystemTimeAsFileTime(&time);
  detail::TimeSpec tp = detail::filetime_to_timespec(time);
  return time_point(__secs(tp.tv_sec) +
                    chrono::duration_cast<duration>(__nsecs(tp.tv_nsec)));
#elif defined(CLOCK_REALTIME)
  typedef chrono::duration<rep, nano> __nsecs;
  struct timespec tp;
  if (0 != clock_gettime(CLOCK_REALTIME, &tp))
    __throw_system_error(errno, "clock_gettime(CLOCK_REALTIME) failed");
  return time_point(__secs(tp.tv_sec) +
                    chrono::duration_cast<duration>(__nsecs(tp.tv_nsec)));
#else
  typedef chrono::duration<rep, micro> __microsecs;
  timeval tv;
  gettimeofday(&tv, 0);
  return time_point(__secs(tv.tv_sec) + __microsecs(tv.tv_usec));
#endif // CLOCK_REALTIME
}

_LIBCPP_END_NAMESPACE_FILESYSTEM
