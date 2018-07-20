//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/filesystem>

// typedef TrivialClock file_time_type;

// RUN: %build -I%libcxx_src_root/src/experimental/filesystem
// RUN: %run

#include <experimental/filesystem>
#include <chrono>
#include <type_traits>
#include <limits>
#include <cstddef>
#include <cassert>

#include "filesystem_common.h"

using namespace std::chrono;
namespace fs = std::experimental::filesystem;
using fs::file_time_type;
using fs::detail::fs_time_util;

enum TestKind { TK_64Bit, TK_32Bit, TK_FloatingPoint };

template <class FileTimeT, class TimeT, class TimeSpec>
constexpr TestKind getTestKind() {
  if (sizeof(TimeT) == 8 && !std::is_floating_point<TimeT>::value)
    return TK_64Bit;
  else if (sizeof(TimeT) == 4 && !std::is_floating_point<TimeT>::value)
    return TK_32Bit;
  else if (std::is_floating_point<TimeT>::value)
    return TK_FloatingPoint;
  else
    assert(false && "test kind not supported");
}

template <class FileTimeT, class TimeT, class TimeSpecT,
          class Base = fs_time_util<FileTimeT, TimeT, TimeSpecT>,
          TestKind = getTestKind<FileTimeT, TimeT, TimeSpecT>()>
struct check_is_representable;

template <class FileTimeT, class TimeT, class TimeSpecT, class Base>
struct check_is_representable<FileTimeT, TimeT, TimeSpecT, Base, TK_64Bit>
    : public Base {

  using Base::convert_timespec;
  using Base::is_representable;
  using Base::max_nsec;
  using Base::max_seconds;
  using Base::min_nsec_timespec;
  using Base::min_seconds;

  static constexpr auto max_time_t = std::numeric_limits<TimeT>::max();
  static constexpr auto min_time_t = std::numeric_limits<TimeT>::min();

  static constexpr bool test_timespec() {
    static_assert(is_representable(TimeSpecT{max_seconds, max_nsec}), "");
    static_assert(!is_representable(TimeSpecT{max_seconds + 1, 0}), "");
    static_assert(!is_representable(TimeSpecT{max_seconds, max_nsec + 1}), "");
    static_assert(!is_representable(TimeSpecT{max_time_t, 0}), "");
    static_assert(is_representable(TimeSpecT{min_seconds, 0}), "");
    static_assert(
        is_representable(TimeSpecT{min_seconds - 1, min_nsec_timespec}), "");
    static_assert(
        is_representable(TimeSpecT{min_seconds - 1, min_nsec_timespec + 1}),
        "");
    static_assert(
        !is_representable(TimeSpecT{min_seconds - 1, min_nsec_timespec - 1}),
        "");
    static_assert(!is_representable(TimeSpecT{min_time_t, 999999999}), "");
    return true;
  }

  static constexpr bool test_file_time_type() {
    static_assert(Base::is_representable(FileTimeT::max()), "");
    static_assert(Base::is_representable(FileTimeT::min()), "");
    return true;
  }

  static constexpr bool test_convert_timespec() {
    static_assert(convert_timespec(TimeSpecT{max_seconds, max_nsec}) ==
                      FileTimeT::max(),
                  "");
    static_assert(convert_timespec(TimeSpecT{max_seconds, max_nsec - 1}) <
                      FileTimeT::max(),
                  "");
    static_assert(convert_timespec(TimeSpecT{max_seconds - 1, 999999999}) <
                      FileTimeT::max(),
                  "");
    static_assert(convert_timespec(TimeSpecT{
                      min_seconds - 1, min_nsec_timespec}) == FileTimeT::min(),
                  "");
    static_assert(
        convert_timespec(TimeSpecT{min_seconds - 1, min_nsec_timespec + 1}) >
            FileTimeT::min(),
        "");
    static_assert(
        convert_timespec(TimeSpecT{min_seconds, 0}) > FileTimeT::min(), "");
    return true;
  }

  static bool test() {
    static_assert(test_timespec(), "");
    static_assert(test_file_time_type(), "");
    static_assert(test_convert_timespec(), "");
    return true;
  }
};

template <class FileTimeT, class TimeT, class TimeSpecT, class Base>
struct check_is_representable<FileTimeT, TimeT, TimeSpecT, Base, TK_32Bit>
    : public Base {
  static constexpr auto max_time_t = std::numeric_limits<TimeT>::max();
  static constexpr auto min_time_t = std::numeric_limits<TimeT>::min();

  using Base::convert_timespec;
  using Base::is_representable;
  using Base::max_nsec;
  using Base::max_seconds;
  using Base::min_nsec_timespec;
  using Base::min_seconds;

  static constexpr bool test_timespec() {
    static_assert(is_representable(TimeSpecT{max_time_t, 999999999}), "");
    static_assert(is_representable(TimeSpecT{max_time_t, 1000000000}), "");
    static_assert(is_representable(TimeSpecT{min_time_t, 0}), "");
    return true;
  }

  static constexpr bool test_file_time_type() {
    static_assert(!is_representable(FileTimeT::max()), "");
    static_assert(!is_representable(FileTimeT::min()), "");
    static_assert(is_representable(FileTimeT(seconds(max_time_t))), "");
    static_assert(is_representable(FileTimeT(seconds(min_time_t))), "");
    return true;
  }

  static constexpr bool test_convert_timespec() {
    // FIXME add tests for 32 bit builds
    return true;
  }

  static bool test() {
    static_assert(test_timespec(), "");
    static_assert(test_file_time_type(), "");
    static_assert(test_convert_timespec(), "");
    return true;
  }
};

template <class FileTimeT, class TimeT, class TimeSpec, class Base>
struct check_is_representable<FileTimeT, TimeT, TimeSpec, Base,
                              TK_FloatingPoint> : public Base {

  static bool test() { return true; }
};

template <class TimeT, class NSecT = long>
struct TestTimeSpec {
  TimeT tv_sec;
  NSecT tv_nsec;
};

template <class Dur>
struct TestClock {
  typedef Dur duration;
  typedef typename duration::rep rep;
  typedef typename duration::period period;
  typedef std::chrono::time_point<TestClock> time_point;
  static constexpr const bool is_steady = false;

  static time_point now() noexcept { return {}; }
};

template <class IntType, class Dur = duration<IntType, std::micro> >
using TestFileTimeT = time_point<TestClock<Dur> >;

int main() {
  assert((
      check_is_representable<file_time_type, time_t, struct timespec>::test()));
  assert((check_is_representable<TestFileTimeT<int64_t>, int64_t,
                                 TestTimeSpec<int64_t, long> >::test()));
  assert((check_is_representable<TestFileTimeT<long long>, int32_t,
                                 TestTimeSpec<int32_t, int32_t> >::test()));

  // Test that insane platforms like ppc64 linux, which use long double as time_t,
  // at least compile.
  assert((check_is_representable<TestFileTimeT<long double>, double,
                                 TestTimeSpec<long double, long> >::test()));
}
