//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-create-symlinks
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: availability-filesystem-missing

// The string reported on errors changed, which makes those tests fail when run
// against already-released libc++'s.
// XFAIL: stdlib=system && target={{.+}}-apple-macosx{{10.15|11.0}}

// <filesystem>

// file_time_type last_write_time(const path& p);
// file_time_type last_write_time(const path& p, std::error_code& ec) noexcept;
// void last_write_time(const path& p, file_time_type new_time);
// void last_write_time(const path& p, file_time_type new_type,
//                      std::error_code& ec) noexcept;

#include <filesystem>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <ratio>
#include <type_traits>

#include "test_macros.h"
#include "filesystem_test_helper.h"

#include <fcntl.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <sys/stat.h>
#endif
namespace fs = std::filesystem;
using namespace fs;

using Sec = std::chrono::duration<file_time_type::rep>;
using Hours = std::chrono::hours;
using Minutes = std::chrono::minutes;
using MilliSec = std::chrono::duration<file_time_type::rep, std::milli>;
using MicroSec = std::chrono::duration<file_time_type::rep, std::micro>;
using NanoSec = std::chrono::duration<file_time_type::rep, std::nano>;
using std::chrono::duration_cast;

#ifdef _WIN32
struct TimeSpec {
  std::int64_t tv_sec;
  std::int64_t tv_nsec;
};
struct StatT {
  TimeSpec st_atim;
  TimeSpec st_mtim;
};
// There were 369 years and 89 leap days from the Windows epoch
// (1601) to the Unix epoch (1970).
#define FILE_TIME_OFFSET_SECS (std::uint64_t(369 * 365 + 89) * (24 * 60 * 60))
static TimeSpec filetime_to_timespec(LARGE_INTEGER li) {
  TimeSpec ret;
  ret.tv_sec = li.QuadPart / 10000000 - FILE_TIME_OFFSET_SECS;
  ret.tv_nsec = (li.QuadPart % 10000000) * 100;
  return ret;
}
static int stat_file(const char *path, StatT *buf, int flags) {
  HANDLE h = CreateFileA(path, FILE_READ_ATTRIBUTES,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_EXISTING,
                         FILE_FLAG_BACKUP_SEMANTICS | flags, nullptr);
  if (h == INVALID_HANDLE_VALUE)
    return -1;
  int ret = -1;
  FILE_BASIC_INFO basic;
  if (GetFileInformationByHandleEx(h, FileBasicInfo, &basic, sizeof(basic))) {
    buf->st_mtim = filetime_to_timespec(basic.LastWriteTime);
    buf->st_atim = filetime_to_timespec(basic.LastAccessTime);
    ret = 0;
  }
  CloseHandle(h);
  return ret;
}
static int stat(const char *path, StatT *buf) {
  return stat_file(path, buf, 0);
}
static int lstat(const char *path, StatT *buf) {
  return stat_file(path, buf, FILE_FLAG_OPEN_REPARSE_POINT);
}
#elif defined(_AIX)
using TimeSpec = st_timespec_t;
using StatT = struct stat;
#else
using TimeSpec = timespec;
using StatT = struct stat;
#endif

#if defined(__APPLE__)
TimeSpec extract_mtime(StatT const& st) { return st.st_mtimespec; }
TimeSpec extract_atime(StatT const& st) { return st.st_atimespec; }
#else
TimeSpec extract_mtime(StatT const& st) { return st.st_mtim; }
TimeSpec extract_atime(StatT const& st) { return st.st_atim; }
#endif

bool ConvertToTimeSpec(TimeSpec& ts, file_time_type ft) {
  using SecFieldT = decltype(TimeSpec::tv_sec);
  using NSecFieldT = decltype(TimeSpec::tv_nsec);
  using SecLim = std::numeric_limits<SecFieldT>;
  using NSecLim = std::numeric_limits<NSecFieldT>;

  auto secs = duration_cast<Sec>(ft.time_since_epoch());
  auto nsecs = duration_cast<NanoSec>(ft.time_since_epoch() - secs);
  if (nsecs.count() < 0) {
    if (Sec::min().count() > SecLim::min()) {
      secs += Sec(1);
      nsecs -= Sec(1);
    } else {
      nsecs = NanoSec(0);
    }
  }
  if (SecLim::max() < secs.count() || SecLim::min() > secs.count())
    return false;
  if (NSecLim::max() < nsecs.count() || NSecLim::min() > nsecs.count())
    return false;
  ts.tv_sec = secs.count();
  ts.tv_nsec = nsecs.count();
  return true;
}

bool ConvertFromTimeSpec(file_time_type& ft, TimeSpec ts) {
  auto secs_part = duration_cast<file_time_type::duration>(Sec(ts.tv_sec));
  if (duration_cast<Sec>(secs_part).count() != ts.tv_sec)
    return false;
  auto subsecs = duration_cast<file_time_type::duration>(NanoSec(ts.tv_nsec));
  auto dur = secs_part + subsecs;
  if (dur < secs_part && subsecs.count() >= 0)
    return false;
  ft = file_time_type(dur);
  return true;
}

bool CompareTimeExact(TimeSpec ts, TimeSpec ts2) {
  return ts2.tv_sec == ts.tv_sec && ts2.tv_nsec == ts.tv_nsec;
}
bool CompareTimeExact(file_time_type ft, TimeSpec ts) {
  TimeSpec ts2 = {};
  if (!ConvertToTimeSpec(ts2, ft))
    return false;
  return CompareTimeExact(ts, ts2);
}
bool CompareTimeExact(TimeSpec ts, file_time_type ft) {
  return CompareTimeExact(ft, ts);
}

struct Times {
  TimeSpec access, write;
};

Times GetTimes(path const& p) {
    StatT st;
    if (::stat(p.string().c_str(), &st) == -1) {
        std::error_code ec(errno, std::generic_category());
#ifndef TEST_HAS_NO_EXCEPTIONS
        throw ec;
#else
        std::exit(EXIT_FAILURE);
#endif
    }
    return {extract_atime(st), extract_mtime(st)};
}

TimeSpec LastAccessTime(path const& p) { return GetTimes(p).access; }

TimeSpec LastWriteTime(path const& p) { return GetTimes(p).write; }

Times GetSymlinkTimes(path const& p) {
  StatT st;
  if (::lstat(p.string().c_str(), &st) == -1) {
    std::error_code ec(errno, std::generic_category());
#ifndef TEST_HAS_NO_EXCEPTIONS
        throw ec;
#else
        std::exit(EXIT_FAILURE);
#endif
    }
    Times res;
    res.access = extract_atime(st);
    res.write = extract_mtime(st);
    return res;
}

namespace {

// In some configurations, the comparison is tautological and the test is valid.
// We disable the warning so that we can actually test it regardless.
TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wtautological-constant-compare")

static const bool SupportsNegativeTimes = [] {
  using namespace std::chrono;
  std::error_code ec;
  TimeSpec old_write_time, new_write_time;
  { // WARNING: Do not assert in this scope.
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    old_write_time = LastWriteTime(file);
    file_time_type tp(seconds(-5));
    fs::last_write_time(file, tp, ec);
    new_write_time = LastWriteTime(file);
  }

  return !ec && new_write_time.tv_sec < 0;
}();

static const bool SupportsMaxTime = [] {
  using namespace std::chrono;
  TimeSpec max_ts = {};
  if (!ConvertToTimeSpec(max_ts, file_time_type::max()))
    return false;

  std::error_code ec;
  TimeSpec old_write_time, new_write_time;
  { // WARNING: Do not assert in this scope.
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    old_write_time = LastWriteTime(file);
    file_time_type tp = file_time_type::max();
    fs::last_write_time(file, tp, ec);
    new_write_time = LastWriteTime(file);
  }
  return !ec && new_write_time.tv_sec > max_ts.tv_sec - 1;
}();

static const bool SupportsMinTime = [] {
  using namespace std::chrono;
  TimeSpec min_ts = {};
  if (!ConvertToTimeSpec(min_ts, file_time_type::min()))
    return false;
  std::error_code ec;
  TimeSpec old_write_time, new_write_time;
  { // WARNING: Do not assert in this scope.
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    old_write_time = LastWriteTime(file);
    file_time_type tp = file_time_type::min();
    fs::last_write_time(file, tp, ec);
    new_write_time = LastWriteTime(file);
  }
  return !ec && new_write_time.tv_sec < min_ts.tv_sec + 1;
}();

static const bool SupportsNanosecondRoundTrip = [] {
  NanoSec ns(3);
  static_assert(std::is_same<file_time_type::period, std::nano>::value, "");

  // Test that the system call we use to set the times also supports nanosecond
  // resolution. (utimes does not)
  file_time_type ft(ns);
  {
    scoped_test_env env;
    const path p = env.create_file("file", 42);
    last_write_time(p, ft);
    return last_write_time(p) == ft;
  }
}();

// The HFS+ filesystem (used by default before macOS 10.13) stores timestamps at
// a 1-second granularity, and APFS (now the default) at a 1 nanosecond granularity.
// 1-second granularity is also the norm on many of the supported filesystems
// on Linux as well.
static const bool WorkaroundStatTruncatesToSeconds = [] {
  MicroSec micros(3);
  static_assert(std::is_same<file_time_type::period, std::nano>::value, "");

  file_time_type ft(micros);
  {
    scoped_test_env env;
    const path p = env.create_file("file", 42);
    if (LastWriteTime(p).tv_nsec != 0)
      return false;
    last_write_time(p, ft);
    return last_write_time(p) != ft && LastWriteTime(p).tv_nsec == 0;
  }
}();

static const bool SupportsMinRoundTrip = [] {
  TimeSpec ts = {};
  if (!ConvertToTimeSpec(ts, file_time_type::min()))
    return false;
  file_time_type min_val = {};
  if (!ConvertFromTimeSpec(min_val, ts))
    return false;
  return min_val == file_time_type::min();
}();

} // end namespace

static bool CompareTime(TimeSpec t1, TimeSpec t2) {
  if (SupportsNanosecondRoundTrip)
    return CompareTimeExact(t1, t2);
  if (t1.tv_sec != t2.tv_sec)
    return false;

  auto diff = std::abs(t1.tv_nsec - t2.tv_nsec);
  if (WorkaroundStatTruncatesToSeconds)
   return diff < duration_cast<NanoSec>(Sec(1)).count();
  return diff < duration_cast<NanoSec>(MicroSec(1)).count();
}

static bool CompareTime(file_time_type t1, TimeSpec t2) {
  TimeSpec ts1 = {};
  if (!ConvertToTimeSpec(ts1, t1))
    return false;
  return CompareTime(ts1, t2);
}

static bool CompareTime(TimeSpec t1, file_time_type t2) {
  return CompareTime(t2, t1);
}

static bool CompareTime(file_time_type t1, file_time_type t2) {
  auto min_secs = duration_cast<Sec>(file_time_type::min().time_since_epoch());
  bool IsMin =
      t1.time_since_epoch() < min_secs || t2.time_since_epoch() < min_secs;

  if (SupportsNanosecondRoundTrip && (!IsMin || SupportsMinRoundTrip))
    return t1 == t2;
  if (IsMin) {
    return duration_cast<Sec>(t1.time_since_epoch()) ==
           duration_cast<Sec>(t2.time_since_epoch());
  }
  file_time_type::duration dur;
  if (t1 > t2)
    dur = t1 - t2;
  else
    dur = t2 - t1;
  if (WorkaroundStatTruncatesToSeconds)
    return duration_cast<Sec>(dur).count() == 0;
  return duration_cast<MicroSec>(dur).count() == 0;
}

// Check if a time point is representable on a given filesystem. Check that:
// (A) 'tp' is representable as a time_t
// (B) 'tp' is non-negative or the filesystem supports negative times.
// (C) 'tp' is not 'file_time_type::max()' or the filesystem supports the max
//     value.
// (D) 'tp' is not 'file_time_type::min()' or the filesystem supports the min
//     value.
inline bool TimeIsRepresentableByFilesystem(file_time_type tp) {
  TimeSpec ts = {};
  if (!ConvertToTimeSpec(ts, tp))
    return false;
  else if (tp.time_since_epoch().count() < 0 && !SupportsNegativeTimes)
    return false;
  else if (tp == file_time_type::max() && !SupportsMaxTime)
    return false;
  else if (tp == file_time_type::min() && !SupportsMinTime)
    return false;
  return true;
}

TEST_DIAGNOSTIC_POP

// Create a sub-second duration using the smallest period the filesystem supports.
file_time_type::duration SubSec(long long val) {
  using SubSecT = file_time_type::duration;
  if (SupportsNanosecondRoundTrip) {
    return duration_cast<SubSecT>(NanoSec(val));
  } else {
    return duration_cast<SubSecT>(MicroSec(val));
  }
}

static void signature_test()
{
    const file_time_type t;
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(last_write_time(p)), file_time_type);
    ASSERT_SAME_TYPE(decltype(last_write_time(p, ec)), file_time_type);
    ASSERT_SAME_TYPE(decltype(last_write_time(p, t)), void);
    ASSERT_SAME_TYPE(decltype(last_write_time(p, t, ec)), void);
    ASSERT_NOT_NOEXCEPT(last_write_time(p));
    ASSERT_NOT_NOEXCEPT(last_write_time(p, t));
    ASSERT_NOEXCEPT(last_write_time(p, ec));
    ASSERT_NOEXCEPT(last_write_time(p, t, ec));
}

static void read_last_write_time_static_env_test()
{
    static_test_env static_env;
    using C = file_time_type::clock;
    file_time_type min = file_time_type::min();
    // Sleep a little to make sure that static_env.File created above is
    // strictly older than C::now() even with a coarser clock granularity
    // in C::now(). (GetSystemTimeAsFileTime on windows has a fairly coarse
    // granularity.)
    SleepFor(MilliSec(30));
    {
        file_time_type ret = last_write_time(static_env.File);
        assert(ret != min);
        assert(ret < C::now());
        assert(CompareTime(ret, LastWriteTime(static_env.File)));

        file_time_type ret2 = last_write_time(static_env.SymlinkToFile);
        assert(CompareTime(ret, ret2));
        assert(CompareTime(ret2, LastWriteTime(static_env.SymlinkToFile)));
    }
    {
        file_time_type ret = last_write_time(static_env.Dir);
        assert(ret != min);
        assert(ret < C::now());
        assert(CompareTime(ret, LastWriteTime(static_env.Dir)));

        file_time_type ret2 = last_write_time(static_env.SymlinkToDir);
        assert(CompareTime(ret, ret2));
        assert(CompareTime(ret2, LastWriteTime(static_env.SymlinkToDir)));
    }
}

static void get_last_write_time_dynamic_env_test()
{
    scoped_test_env env;

    const path file = env.create_file("file", 42);
    const path dir = env.create_dir("dir");

    const auto file_times = GetTimes(file);
    const TimeSpec file_write_time = file_times.write;
    const auto dir_times = GetTimes(dir);
    const TimeSpec dir_write_time = dir_times.write;

    file_time_type ftime = last_write_time(file);
    assert(CompareTime(ftime, file_write_time));

    file_time_type dtime = last_write_time(dir);
    assert(CompareTime(dtime, dir_write_time));

    SleepFor(std::chrono::seconds(2));

    // update file and add a file to the directory. Make sure the times increase.
    std::FILE* of = std::fopen(file.string().c_str(), "a");
    std::fwrite("hello", 1, sizeof("hello"), of);
    std::fclose(of);
    env.create_file("dir/file1", 1);

    file_time_type ftime2 = last_write_time(file);
    file_time_type dtime2 = last_write_time(dir);

    assert(ftime2 > ftime);
    assert(dtime2 > dtime);
    assert(CompareTime(LastWriteTime(file), ftime2));
    assert(CompareTime(LastWriteTime(dir), dtime2));
}


static void set_last_write_time_dynamic_env_test()
{
    using Clock = file_time_type::clock;
    scoped_test_env env;

    const path file = env.create_file("file", 42);
    const path dir = env.create_dir("dir");
    const auto now = Clock::now();
    const file_time_type epoch_time = now - now.time_since_epoch();

    const file_time_type future_time = now + Hours(3) + Sec(42) + SubSec(17);
    const file_time_type past_time = now - Minutes(3) - Sec(42) - SubSec(17);
    const file_time_type before_epoch_time =
        epoch_time - Minutes(3) - Sec(42) - SubSec(17);
    (void)before_epoch_time;
    // FreeBSD has a bug in their utimes implementation where the time is not update
    // when the number of seconds is '-1'.
#if defined(__FreeBSD__) || defined(__NetBSD__)
    const file_time_type just_before_epoch_time =
        epoch_time - Sec(2) - SubSec(17);
#else
    const file_time_type just_before_epoch_time = epoch_time - SubSec(17);
    (void)just_before_epoch_time;
#endif

    struct TestCase {
      const char * case_name;
      path p;
      file_time_type new_time;
    } cases[] = {
        {"file, epoch_time", file, epoch_time},
        {"dir, epoch_time", dir, epoch_time},
        {"file, future_time", file, future_time},
        {"dir, future_time", dir, future_time},
        {"file, past_time", file, past_time},
        {"dir, past_time", dir, past_time}
        // Exclude file time types of before epoch time from testing on AIX
        // because AIX system call utimensat() does not accept the times
        // parameter having a negative tv_sec or tv_nsec field.
#if !defined(_AIX)
        ,
        {"file, before_epoch_time", file, before_epoch_time},
        {"dir, before_epoch_time", dir, before_epoch_time},
        {"file, just_before_epoch_time", file, just_before_epoch_time},
        {"dir, just_before_epoch_time", dir, just_before_epoch_time}
#endif
    };

    for (const auto& TC : cases) {
        const auto old_times = GetTimes(TC.p);
        file_time_type old_time;
        assert(ConvertFromTimeSpec(old_time, old_times.write));

        std::error_code ec = GetTestEC();
        last_write_time(TC.p, TC.new_time, ec);
        assert(!ec);

        ec = GetTestEC();
        file_time_type  got_time = last_write_time(TC.p, ec);
        assert(!ec);

        if (TimeIsRepresentableByFilesystem(TC.new_time)) {
            assert(got_time != old_time);
            assert(CompareTime(got_time, TC.new_time));
            assert(CompareTime(LastAccessTime(TC.p), old_times.access));
        }
    }
}

static void last_write_time_symlink_test()
{
    using Clock = file_time_type::clock;

    scoped_test_env env;

    const path file = env.create_file("file", 42);
    const path sym = env.create_symlink("file", "sym");

    const file_time_type new_time = Clock::now() + Hours(3);

    const auto old_times = GetTimes(sym);
    const auto old_sym_times = GetSymlinkTimes(sym);

    std::error_code ec = GetTestEC();
    last_write_time(sym, new_time, ec);
    assert(!ec);

    file_time_type  got_time = last_write_time(sym);
    assert(!CompareTime(got_time, old_times.write));
    if (!WorkaroundStatTruncatesToSeconds) {
      assert(got_time == new_time);
    } else {
      assert(CompareTime(got_time, new_time));
    }

    assert(CompareTime(LastWriteTime(file), new_time));
    assert(CompareTime(LastAccessTime(sym), old_times.access));
    Times sym_times = GetSymlinkTimes(sym);
    assert(CompareTime(sym_times.write, old_sym_times.write));
}


static void test_write_min_time()
{
    scoped_test_env env;
    const path p = env.create_file("file", 42);
    const file_time_type old_time = last_write_time(p);
    file_time_type new_time = file_time_type::min();

    std::error_code ec = GetTestEC();
    last_write_time(p, new_time, ec);
    file_time_type tt = last_write_time(p);

    if (TimeIsRepresentableByFilesystem(new_time)) {
        assert(!ec);
        assert(CompareTime(tt, new_time));

        last_write_time(p, old_time);
        new_time = file_time_type::min() + SubSec(1);

        ec = GetTestEC();
        last_write_time(p, new_time, ec);
        tt = last_write_time(p);

        if (TimeIsRepresentableByFilesystem(new_time)) {
            assert(!ec);
            assert(CompareTime(tt, new_time));
        } else {
          assert(ErrorIs(ec, std::errc::value_too_large));
          assert(tt == old_time);
        }
    } else {
      assert(ErrorIs(ec, std::errc::value_too_large));
      assert(tt == old_time);
    }
}

static void test_write_max_time() {
  scoped_test_env env;
  const path p = env.create_file("file", 42);
  const file_time_type old_time = last_write_time(p);
  file_time_type new_time = file_time_type::max();

  std::error_code ec = GetTestEC();
  last_write_time(p, new_time, ec);
  file_time_type tt = last_write_time(p);

  if (TimeIsRepresentableByFilesystem(new_time)) {
    assert(!ec);
    assert(CompareTime(tt, new_time));
  } else {
    assert(ErrorIs(ec, std::errc::value_too_large));
    assert(tt == old_time);
  }
}

static void test_value_on_failure()
{
    static_test_env static_env;
    const path p = static_env.DNE;
    std::error_code ec = GetTestEC();
    assert(last_write_time(p, ec) == file_time_type::min());
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));
}

// Windows doesn't support setting perms::none to trigger failures
// reading directories.
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
static void test_exists_fails()
{
    scoped_test_env env;
    const path dir = env.create_dir("dir");
    const path file = env.create_file("dir/file", 42);
    permissions(dir, perms::none);

    std::error_code ec = GetTestEC();
    assert(last_write_time(file, ec) == file_time_type::min());
    assert(ErrorIs(ec, std::errc::permission_denied));

    ExceptionChecker Checker(file, std::errc::permission_denied,
                             "last_write_time");
    TEST_VALIDATE_EXCEPTION(filesystem_error, Checker, last_write_time(file));
}
#endif // TEST_WIN_NO_FILESYSTEM_PERMS_NONE

int main(int, char**) {
    signature_test();
    read_last_write_time_static_env_test();
    get_last_write_time_dynamic_env_test();
    set_last_write_time_dynamic_env_test();
    last_write_time_symlink_test();
    test_write_min_time();
    test_write_max_time();
    test_value_on_failure();
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
    test_exists_fails();
#endif
  return 0;
}
