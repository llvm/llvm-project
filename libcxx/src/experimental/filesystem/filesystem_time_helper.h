//===----------------------------------------------------------------------===////
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===////

#ifndef FILESYSTEM_TIME_HELPER_H
#define FILESYSTEM_TIME_HELPER_H

#include "experimental/__config"
#include "chrono"
#include "cstdlib"
#include "climits"

#include <unistd.h>
#include <sys/stat.h>
#if !defined(UTIME_OMIT)
#include <sys/time.h> // for ::utimes as used in __last_write_time
#endif

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_FILESYSTEM

namespace time_detail { namespace {

using namespace chrono;

template <class FileTimeT,
          bool IsFloat = is_floating_point<typename FileTimeT::rep>::value>
struct fs_time_util_base {
  static constexpr auto max_seconds =
      duration_cast<seconds>(FileTimeT::duration::max()).count();

  static constexpr auto max_nsec =
      duration_cast<nanoseconds>(FileTimeT::duration::max() -
                                 seconds(max_seconds))
          .count();

  static constexpr auto min_seconds =
      duration_cast<seconds>(FileTimeT::duration::min()).count();

  static constexpr auto min_nsec_timespec =
      duration_cast<nanoseconds>(
          (FileTimeT::duration::min() - seconds(min_seconds)) + seconds(1))
          .count();

  // Static assert that these values properly round trip.
  static_assert((seconds(min_seconds) +
                 duration_cast<microseconds>(nanoseconds(min_nsec_timespec))) -
                        duration_cast<microseconds>(seconds(1)) ==
                    FileTimeT::duration::min(),
                "");
};

template <class FileTimeT>
struct fs_time_util_base<FileTimeT, true> {
  static const long long max_seconds;
  static const long long max_nsec;
  static const long long min_seconds;
  static const long long min_nsec_timespec;
};

template <class FileTimeT>
const long long fs_time_util_base<FileTimeT, true>::max_seconds =
    duration_cast<seconds>(FileTimeT::duration::max()).count();

template <class FileTimeT>
const long long fs_time_util_base<FileTimeT, true>::max_nsec =
    duration_cast<nanoseconds>(FileTimeT::duration::max() -
                               seconds(max_seconds))
        .count();

template <class FileTimeT>
const long long fs_time_util_base<FileTimeT, true>::min_seconds =
    duration_cast<seconds>(FileTimeT::duration::min()).count();

template <class FileTimeT>
const long long fs_time_util_base<FileTimeT, true>::min_nsec_timespec =
    duration_cast<nanoseconds>((FileTimeT::duration::min() -
                                seconds(min_seconds)) +
                               seconds(1))
        .count();

template <class FileTimeT, class TimeT, class TimeSpecT>
struct fs_time_util : fs_time_util_base<FileTimeT> {
  using Base = fs_time_util_base<FileTimeT>;
  using Base::max_nsec;
  using Base::max_seconds;
  using Base::min_nsec_timespec;
  using Base::min_seconds;

public:
  template <class CType, class ChronoType>
  static bool checked_set(CType* out, ChronoType time) {
    using Lim = numeric_limits<CType>;
    if (time > Lim::max() || time < Lim::min())
      return false;
    *out = static_cast<CType>(time);
    return true;
  }

  static _LIBCPP_CONSTEXPR_AFTER_CXX11 bool is_representable(TimeSpecT tm) {
    if (tm.tv_sec >= 0) {
      return (tm.tv_sec < max_seconds) ||
             (tm.tv_sec == max_seconds && tm.tv_nsec <= max_nsec);
    } else if (tm.tv_sec == (min_seconds - 1)) {
      return tm.tv_nsec >= min_nsec_timespec;
    } else {
      return (tm.tv_sec >= min_seconds);
    }
  }

  static _LIBCPP_CONSTEXPR_AFTER_CXX11 bool is_representable(FileTimeT tm) {
    auto secs = duration_cast<seconds>(tm.time_since_epoch());
    auto nsecs = duration_cast<nanoseconds>(tm.time_since_epoch() - secs);
    if (nsecs.count() < 0) {
      secs = secs + seconds(1);
      nsecs = nsecs + seconds(1);
    }
    using TLim = numeric_limits<TimeT>;
    if (secs.count() >= 0)
      return secs.count() <= TLim::max();
    return secs.count() >= TLim::min();
  }

  static _LIBCPP_CONSTEXPR_AFTER_CXX11 FileTimeT
  convert_timespec(TimeSpecT tm) {
    auto adj_msec = duration_cast<microseconds>(nanoseconds(tm.tv_nsec));
    if (tm.tv_sec >= 0) {
      auto Dur = seconds(tm.tv_sec) + microseconds(adj_msec);
      return FileTimeT(Dur);
    } else if (duration_cast<microseconds>(nanoseconds(tm.tv_nsec)).count() ==
               0) {
      return FileTimeT(seconds(tm.tv_sec));
    } else { // tm.tv_sec < 0
      auto adj_subsec =
          duration_cast<microseconds>(seconds(1) - nanoseconds(tm.tv_nsec));
      auto Dur = seconds(tm.tv_sec + 1) - adj_subsec;
      return FileTimeT(Dur);
    }
  }

  template <class SubSecDurT, class SubSecT>
  static bool set_times_checked(TimeT* sec_out, SubSecT* subsec_out,
                                FileTimeT tp) {
    using namespace chrono;
    auto dur = tp.time_since_epoch();
    auto sec_dur = duration_cast<seconds>(dur);
    auto subsec_dur = duration_cast<SubSecDurT>(dur - sec_dur);
    // The tv_nsec and tv_usec fields must not be negative so adjust accordingly
    if (subsec_dur.count() < 0) {
      if (sec_dur.count() > min_seconds) {
        sec_dur -= seconds(1);
        subsec_dur += seconds(1);
      } else {
        subsec_dur = SubSecDurT::zero();
      }
    }
    return checked_set(sec_out, sec_dur.count()) &&
           checked_set(subsec_out, subsec_dur.count());
  }
};

} // end namespace
} // end namespace time_detail

using time_detail::fs_time_util;

_LIBCPP_END_NAMESPACE_EXPERIMENTAL_FILESYSTEM

#endif // FILESYSTEM_TIME_HELPER_H
