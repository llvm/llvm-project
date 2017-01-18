//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

// file_time_type last_write_time(const path& p);
// file_time_type last_write_time(const path& p, std::error_code& ec) noexcept;
// void last_write_time(const path& p, file_time_type new_time);
// void last_write_time(const path& p, file_time_type new_type,
//                      std::error_code& ec) noexcept;


#include <experimental/filesystem>
#include <type_traits>
#include <chrono>
#include <fstream>
#include <cstdlib>

#include "test_macros.h"
#include "rapid-cxx-test.hpp"
#include "filesystem_test_helper.hpp"

#include <sys/stat.h>
#include <iostream>

using namespace std::experimental::filesystem;


std::pair<std::time_t, std::time_t> GetTimes(path const& p) {
    using Clock = file_time_type::clock;
    struct ::stat st;
    if (::stat(p.c_str(), &st) == -1) {
        std::error_code ec(errno, std::generic_category());
#ifndef TEST_HAS_NO_EXCEPTIONS
        throw ec;
#else
        std::cerr << ec.message() << std::endl;
        std::exit(EXIT_FAILURE);
#endif
    }
    return {st.st_atime, st.st_mtime};
}

std::time_t LastAccessTime(path const& p) {
    return GetTimes(p).first;
}

std::time_t LastWriteTime(path const& p) {
    return GetTimes(p).second;
}

std::pair<std::time_t, std::time_t> GetSymlinkTimes(path const& p) {
    using Clock = file_time_type::clock;
    struct ::stat st;
    if (::lstat(p.c_str(), &st) == -1) {
        std::error_code ec(errno, std::generic_category());
#ifndef TEST_HAS_NO_EXCEPTIONS
        throw ec;
#else
        std::cerr << ec.message() << std::endl;
        std::exit(EXIT_FAILURE);
#endif
    }
    return {st.st_atime, st.st_mtime};
}

namespace {
bool TestSupportsNegativeTimes() {
    using namespace std::chrono;
    std::error_code ec;
    std::time_t old_write_time, new_write_time;
    { // WARNING: Do not assert in this scope.
        scoped_test_env env;
        const path file = env.create_file("file", 42);
        old_write_time = LastWriteTime(file);
        file_time_type tp(seconds(-5));
        fs::last_write_time(file, tp, ec);
        new_write_time = LastWriteTime(file);
    }
    return !ec && new_write_time <= -5;
}

bool TestSupportsMaxTime() {
    using namespace std::chrono;
    using Lim = std::numeric_limits<std::time_t>;
    auto max_sec = duration_cast<seconds>(file_time_type::max().time_since_epoch()).count();
    if (max_sec > Lim::max()) return false;
    std::error_code ec;
    std::time_t old_write_time, new_write_time;
    { // WARNING: Do not assert in this scope.
        scoped_test_env env;
        const path file = env.create_file("file", 42);
        old_write_time = LastWriteTime(file);
        file_time_type tp = file_time_type::max();
        fs::last_write_time(file, tp, ec);
        new_write_time = LastWriteTime(file);
    }
    return !ec && new_write_time > max_sec - 1;
}

static const bool SupportsNegativeTimes = TestSupportsNegativeTimes();
static const bool SupportsMaxTime = TestSupportsMaxTime();

} // end namespace

// Check if a time point is representable on a given filesystem. Check that:
// (A) 'tp' is representable as a time_t
// (B) 'tp' is non-negative or the filesystem supports negative times.
// (C) 'tp' is not 'file_time_type::max()' or the filesystem supports the max
//     value.
inline bool TimeIsRepresentableByFilesystem(file_time_type tp) {
    using namespace std::chrono;
    using Lim = std::numeric_limits<std::time_t>;
    auto sec = duration_cast<seconds>(tp.time_since_epoch()).count();
    auto microsec = duration_cast<microseconds>(tp.time_since_epoch()).count();
    if (sec < Lim::min() || sec > Lim::max())   return false;
    else if (microsec < 0 && !SupportsNegativeTimes) return false;
    else if (tp == file_time_type::max() && !SupportsMaxTime) return false;
    return true;
}

TEST_SUITE(exists_test_suite)

TEST_CASE(signature_test)
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

TEST_CASE(read_last_write_time_static_env_test)
{
    using C = file_time_type::clock;
    file_time_type min = file_time_type::min();
    {
        file_time_type ret = last_write_time(StaticEnv::File);
        TEST_CHECK(ret != min);
        TEST_CHECK(ret < C::now());
        TEST_CHECK(C::to_time_t(ret) == LastWriteTime(StaticEnv::File));

        file_time_type ret2 = last_write_time(StaticEnv::SymlinkToFile);
        TEST_CHECK(ret == ret2);
        TEST_CHECK(C::to_time_t(ret2) == LastWriteTime(StaticEnv::SymlinkToFile));
    }
    {
        file_time_type ret = last_write_time(StaticEnv::Dir);
        TEST_CHECK(ret != min);
        TEST_CHECK(ret < C::now());
        TEST_CHECK(C::to_time_t(ret) == LastWriteTime(StaticEnv::Dir));

        file_time_type ret2 = last_write_time(StaticEnv::SymlinkToDir);
        TEST_CHECK(ret == ret2);
        TEST_CHECK(C::to_time_t(ret2) == LastWriteTime(StaticEnv::SymlinkToDir));
    }
}

TEST_CASE(get_last_write_time_dynamic_env_test)
{
    using Clock = file_time_type::clock;
    using Sec = std::chrono::seconds;
    scoped_test_env env;

    const path file = env.create_file("file", 42);
    const path dir = env.create_dir("dir");

    const auto file_times = GetTimes(file);
    const std::time_t file_access_time = file_times.first;
    const std::time_t file_write_time = file_times.second;
    const auto dir_times = GetTimes(dir);
    const std::time_t dir_access_time = dir_times.first;
    const std::time_t dir_write_time = dir_times.second;

    file_time_type ftime = last_write_time(file);
    TEST_CHECK(Clock::to_time_t(ftime) == file_write_time);

    file_time_type dtime = last_write_time(dir);
    TEST_CHECK(Clock::to_time_t(dtime) == dir_write_time);

    SleepFor(Sec(2));

    // update file and add a file to the directory. Make sure the times increase.
    std::ofstream of(file, std::ofstream::app);
    of << "hello";
    of.close();
    env.create_file("dir/file1", 1);

    file_time_type ftime2 = last_write_time(file);
    file_time_type dtime2 = last_write_time(dir);

    TEST_CHECK(ftime2 > ftime);
    TEST_CHECK(dtime2 > dtime);
    TEST_CHECK(LastAccessTime(file) == file_access_time ||
               LastAccessTime(file) == Clock::to_time_t(ftime2));
    TEST_CHECK(LastAccessTime(dir) == dir_access_time);
}


TEST_CASE(set_last_write_time_dynamic_env_test)
{
    using Clock = file_time_type::clock;
    using Sec = std::chrono::seconds;
    using Hours = std::chrono::hours;
    using Minutes = std::chrono::minutes;
    using MicroSec = std::chrono::microseconds;
    scoped_test_env env;

    const path file = env.create_file("file", 42);
    const path dir = env.create_dir("dir");
    const auto now = Clock::now();
    const file_time_type epoch_time = now - now.time_since_epoch();

    const file_time_type future_time = now + Hours(3) + Sec(42) + MicroSec(17);
    const file_time_type past_time = now - Minutes(3) - Sec(42) - MicroSec(17);
    const file_time_type before_epoch_time = epoch_time - Minutes(3) - Sec(42) - MicroSec(17);
    // FreeBSD has a bug in their utimes implementation where the time is not update
    // when the number of seconds is '-1'.
#if defined(__FreeBSD__)
    const file_time_type just_before_epoch_time = epoch_time - Sec(2) - MicroSec(17);
#else
    const file_time_type just_before_epoch_time = epoch_time - MicroSec(17);
#endif

    struct TestCase {
      path p;
      file_time_type new_time;
    } cases[] = {
        {file, epoch_time},
        {dir, epoch_time},
        {file, future_time},
        {dir, future_time},
        {file, past_time},
        {dir, past_time},
        {file, before_epoch_time},
        {dir, before_epoch_time},
        {file, just_before_epoch_time},
        {dir, just_before_epoch_time}
    };
    for (const auto& TC : cases) {
        const auto old_times = GetTimes(TC.p);
        file_time_type old_time(Sec(old_times.second));

        std::error_code ec = GetTestEC();
        last_write_time(TC.p, TC.new_time, ec);
        TEST_CHECK(!ec);

        file_time_type  got_time = last_write_time(TC.p);

        if (TimeIsRepresentableByFilesystem(TC.new_time)) {
            TEST_CHECK(got_time != old_time);
            if (TC.new_time < epoch_time) {
                TEST_CHECK(got_time <= TC.new_time);
                TEST_CHECK(got_time > TC.new_time - Sec(1));
            } else {
                TEST_CHECK(got_time <= TC.new_time + Sec(1));
                TEST_CHECK(got_time >= TC.new_time - Sec(1));
            }
            TEST_CHECK(LastAccessTime(TC.p) == old_times.first);
        }
    }
}

TEST_CASE(last_write_time_symlink_test)
{
    using Clock = file_time_type::clock;
    using Sec = std::chrono::seconds;
    using Hours = std::chrono::hours;
    using Minutes = std::chrono::minutes;

    scoped_test_env env;

    const path file = env.create_file("file", 42);
    const path sym = env.create_symlink("file", "sym");

    const file_time_type new_time = Clock::now() + Hours(3);

    const auto old_times = GetTimes(sym);
    const auto old_sym_times = GetSymlinkTimes(sym);

    std::error_code ec = GetTestEC();
    last_write_time(sym, new_time, ec);
    TEST_CHECK(!ec);

    const std::time_t new_time_t = Clock::to_time_t(new_time);
    file_time_type  got_time = last_write_time(sym);
    std::time_t got_time_t = Clock::to_time_t(got_time);

    TEST_CHECK(got_time_t != old_times.second);
    TEST_CHECK(got_time_t == new_time_t);
    TEST_CHECK(LastWriteTime(file) == new_time_t);
    TEST_CHECK(LastAccessTime(sym) == old_times.first);
    TEST_CHECK(GetSymlinkTimes(sym) == old_sym_times);
}


TEST_CASE(test_write_min_time)
{
    using Clock = file_time_type::clock;
    using Sec = std::chrono::seconds;
    using MicroSec = std::chrono::microseconds;
    using Lim = std::numeric_limits<std::time_t>;
    scoped_test_env env;
    const path p = env.create_file("file", 42);

    std::error_code ec = GetTestEC();
    file_time_type new_time = file_time_type::min();

    last_write_time(p, new_time, ec);
    file_time_type tt = last_write_time(p);

    if (TimeIsRepresentableByFilesystem(new_time)) {
        TEST_CHECK(!ec);
        TEST_CHECK(tt >= new_time);
        TEST_CHECK(tt < new_time + Sec(1));
    }

    ec = GetTestEC();
    last_write_time(p, Clock::now());

    new_time = file_time_type::min() + MicroSec(1);

    last_write_time(p, new_time, ec);
    tt = last_write_time(p);

    if (TimeIsRepresentableByFilesystem(new_time)) {
        TEST_CHECK(!ec);
        TEST_CHECK(tt >= new_time);
        TEST_CHECK(tt < new_time + Sec(1));
    }
}



TEST_CASE(test_write_min_max_time)
{
    using Clock = file_time_type::clock;
    using Sec = std::chrono::seconds;
    using Hours = std::chrono::hours;
    using Lim = std::numeric_limits<std::time_t>;
    scoped_test_env env;
    const path p = env.create_file("file", 42);

    std::error_code ec = GetTestEC();
    file_time_type new_time = file_time_type::max();

    ec = GetTestEC();
    last_write_time(p, new_time, ec);
    file_time_type tt = last_write_time(p);

    if (TimeIsRepresentableByFilesystem(new_time)) {
        TEST_CHECK(!ec);
        TEST_CHECK(tt > new_time - Sec(1));
        TEST_CHECK(tt <= new_time);
    }
}

TEST_CASE(test_value_on_failure)
{
    const path p = StaticEnv::DNE;
    std::error_code ec = GetTestEC();
    TEST_CHECK(last_write_time(p, ec) == file_time_type::min());
    TEST_CHECK(ec);
    TEST_CHECK(ec != GetTestEC());
}

TEST_CASE(test_exists_fails)
{
    scoped_test_env env;
    const path dir = env.create_dir("dir");
    const path file = env.create_file("dir/file", 42);
    permissions(dir, perms::none);

    std::error_code ec = GetTestEC();
    TEST_CHECK(last_write_time(file, ec) == file_time_type::min());
    TEST_CHECK(ec);
    TEST_CHECK(ec != GetTestEC());

    TEST_CHECK_THROW(filesystem_error, last_write_time(file));
}

TEST_SUITE_END()
