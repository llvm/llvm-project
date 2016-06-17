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
#include <thread>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.hpp"
#include "filesystem_test_helper.hpp"

#include <sys/stat.h>

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

    std::this_thread::sleep_for(Sec(2));

    // update file and add a file to the directory. Make sure the times increase.
    std::ofstream of(file, std::ofstream::app);
    of << "hello";
    of.close();
    env.create_file("dir/file1", 1);

    file_time_type ftime2 = last_write_time(file);
    file_time_type dtime2 = last_write_time(dir);

    TEST_CHECK(ftime2 > ftime);
    TEST_CHECK(dtime2 > dtime);
    TEST_CHECK(LastAccessTime(file) == file_access_time);
    TEST_CHECK(LastAccessTime(dir) == dir_access_time);
}


TEST_CASE(set_last_write_time_dynamic_env_test)
{
    using Clock = file_time_type::clock;
    using Sec = std::chrono::seconds;
    using Hours = std::chrono::hours;
    using Minutes = std::chrono::minutes;

    scoped_test_env env;

    const path file = env.create_file("file", 42);
    const path dir = env.create_dir("dir");

    const file_time_type future_time = Clock::now() + Hours(3);
    const file_time_type past_time = Clock::now() - Minutes(3);

    struct TestCase {
      path p;
      file_time_type new_time;
    } cases[] = {
        {file, future_time},
        {dir, future_time},
        {file, past_time},
        {dir, past_time}
    };
    for (const auto& TC : cases) {
        const auto old_times = GetTimes(TC.p);

        std::error_code ec = GetTestEC();
        last_write_time(TC.p, TC.new_time, ec);
        TEST_CHECK(!ec);

        const std::time_t new_time_t = Clock::to_time_t(TC.new_time);
        file_time_type  got_time = last_write_time(TC.p);
        std::time_t got_time_t = Clock::to_time_t(got_time);

        TEST_CHECK(got_time_t != old_times.second);
        TEST_CHECK(got_time_t == new_time_t);
        TEST_CHECK(LastAccessTime(TC.p) == old_times.first);
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

TEST_CASE(test_write_min_max_time)
{
    using Clock = file_time_type::clock;
    using Sec = std::chrono::seconds;
    using Hours = std::chrono::hours;
    scoped_test_env env;
    const path p = env.create_file("file", 42);

    file_time_type last_time = last_write_time(p);

    file_time_type new_time = file_time_type::min();
    std::error_code ec = GetTestEC();
    last_write_time(p, new_time, ec);
    file_time_type tt = last_write_time(p);
    if (ec) {
        TEST_CHECK(ec != GetTestEC());
        TEST_CHECK(tt == last_time);
    } else {
        file_time_type max_allowed = new_time + Sec(1);
        TEST_CHECK(tt >= new_time);
        TEST_CHECK(tt < max_allowed);
    }

    last_time = tt;
    new_time = file_time_type::max();
    ec = GetTestEC();
    last_write_time(p, new_time, ec);

    tt = last_write_time(p);
    if (ec) {
        TEST_CHECK(ec != GetTestEC());
        TEST_CHECK(tt == last_time);
    } else {
        file_time_type min_allowed = new_time - Sec(1);
        TEST_CHECK(tt > min_allowed);
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
