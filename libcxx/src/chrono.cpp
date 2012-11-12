//===------------------------- chrono.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "chrono"
#include <sys/time.h>        //for gettimeofday and timeval
#if __APPLE__
#include <mach/mach_time.h>  // mach_absolute_time, mach_timebase_info_data_t
#else  /* !__APPLE__ */
#include <cerrno>  // errno
#include <system_error>  // __throw_system_error
#include <time.h>  // clock_gettime, CLOCK_MONOTONIC
#endif  // __APPLE__

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono
{

// system_clock

system_clock::time_point
system_clock::now() _NOEXCEPT
{
    timeval tv;
    gettimeofday(&tv, 0);
    return time_point(seconds(tv.tv_sec) + microseconds(tv.tv_usec));
}

time_t
system_clock::to_time_t(const time_point& t) _NOEXCEPT
{
    return time_t(duration_cast<seconds>(t.time_since_epoch()).count());
}

system_clock::time_point
system_clock::from_time_t(time_t t) _NOEXCEPT
{
    return system_clock::time_point(seconds(t));
}

// steady_clock

#if __APPLE__
//   mach_absolute_time() * MachInfo.numer / MachInfo.denom is the number of
//   nanoseconds since the computer booted up.  MachInfo.numer and MachInfo.denom
//   are run time constants supplied by the OS.  This clock has no relationship
//   to the Gregorian calendar.  It's main use is as a high resolution timer.

// MachInfo.numer / MachInfo.denom is often 1 on the latest equipment.  Specialize
//   for that case as an optimization.

#pragma GCC visibility push(hidden)

static
steady_clock::rep
steady_simplified()
{
    return static_cast<steady_clock::rep>(mach_absolute_time());
}

static
double
compute_steady_factor()
{
    mach_timebase_info_data_t MachInfo;
    mach_timebase_info(&MachInfo);
    return static_cast<double>(MachInfo.numer) / MachInfo.denom;
}

static
steady_clock::rep
steady_full()
{
    static const double factor = compute_steady_factor();
    return static_cast<steady_clock::rep>(mach_absolute_time() * factor);
}

typedef steady_clock::rep (*FP)();

static
FP
init_steady_clock()
{
    mach_timebase_info_data_t MachInfo;
    mach_timebase_info(&MachInfo);
    if (MachInfo.numer == MachInfo.denom)
        return &steady_simplified;
    return &steady_full;
}

#pragma GCC visibility pop

steady_clock::time_point
steady_clock::now() _NOEXCEPT
{
    static FP fp = init_steady_clock();
    return time_point(duration(fp()));
}

#else  // __APPLE__
// FIXME: We assume that clock_gettime(CLOCK_MONOTONIC) works on
// non-apple systems.  Instead, we should check _POSIX_TIMERS and
// _POSIX_MONOTONIC_CLOCK and fall back to something else if those
// don't exist.

// Warning:  If this is not truly steady, then it is non-conforming.  It is
//  better for it to not exist and have the rest of libc++ use system_clock
//  instead.

steady_clock::time_point
steady_clock::now() _NOEXCEPT
{
    struct timespec tp;
    if (0 != clock_gettime(CLOCK_MONOTONIC, &tp))
        __throw_system_error(errno, "clock_gettime(CLOCK_MONOTONIC) failed");
    return time_point(seconds(tp.tv_sec) + nanoseconds(tp.tv_nsec));
}
#endif  // __APPLE__

}

_LIBCPP_END_NAMESPACE_STD
