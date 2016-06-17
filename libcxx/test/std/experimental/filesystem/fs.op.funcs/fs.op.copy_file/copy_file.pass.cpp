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

// bool copy_file(const path& from, const path& to);
// bool copy_file(const path& from, const path& to, error_code& ec) noexcept;
// bool copy_file(const path& from, const path& to, copy_options options);
// bool copy_file(const path& from, const path& to, copy_options options,
//           error_code& ec) noexcept;

#include <experimental/filesystem>
#include <type_traits>
#include <chrono>
#include <thread>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.hpp"
#include "filesystem_test_helper.hpp"

using namespace std::experimental::filesystem;
namespace fs = std::experimental::filesystem;

using CO = fs::copy_options;

TEST_SUITE(filesystem_copy_file_test_suite)

TEST_CASE(test_signatures)
{
    const path p; ((void)p);
    const copy_options opts{}; ((void)opts);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(fs::copy_file(p, p)), bool);
    ASSERT_SAME_TYPE(decltype(fs::copy_file(p, p, opts)), bool);
    ASSERT_SAME_TYPE(decltype(fs::copy_file(p, p, ec)), bool);
    ASSERT_SAME_TYPE(decltype(fs::copy_file(p, p, opts, ec)), bool);
    ASSERT_NOT_NOEXCEPT(fs::copy_file(p, p));
    ASSERT_NOT_NOEXCEPT(fs::copy_file(p, p, opts));
    ASSERT_NOEXCEPT(fs::copy_file(p, p, ec));
    ASSERT_NOEXCEPT(fs::copy_file(p, p, opts, ec));
}

TEST_CASE(test_error_reporting)
{
    auto checkThrow = [](path const& f, path const& t, const std::error_code& ec)
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            fs::copy_file(f, t);
            return false;
        } catch (filesystem_error const& err) {
            return err.path1() == f
                && err.path2() == t
                && err.code() == ec;
        }
#else
        return true;
#endif
    };

    scoped_test_env env;
    const path file = env.create_file("file1", 42);
    const path file2 = env.create_file("file2", 55);
    const path dne = env.make_env_path("dne");
    { // exists(to) && equivalent(to, from)
        std::error_code ec;
        TEST_CHECK(fs::copy_file(file, file, ec) == false);
        TEST_REQUIRE(ec);
        TEST_CHECK(checkThrow(file, file, ec));
    }
    { // exists(to) && !(skip_existing | overwrite_existing | update_existing)
        std::error_code ec;
        TEST_CHECK(fs::copy_file(file, file2, ec) == false);
        TEST_REQUIRE(ec);
        TEST_CHECK(checkThrow(file, file2, ec));
    }
}

TEST_CASE(copy_file)
{
    scoped_test_env env;
    const path file = env.create_file("file1", 42);

    { // !exists(to)
        const path dest = env.make_env_path("dest1");
        std::error_code ec;
        TEST_REQUIRE(fs::copy_file(file, dest, ec) == true);
        TEST_CHECK(!ec);
        TEST_CHECK(file_size(dest) == 42);
    }
    { // exists(to) && overwrite_existing
        const path dest = env.create_file("dest2", 55);
        std::error_code ec;
        TEST_REQUIRE(fs::copy_file(file, dest,
                                   copy_options::overwrite_existing, ec) == true);
        TEST_CHECK(!ec);
        TEST_CHECK(file_size(dest) == 42);
    }
    { // exists(to) && update_existing
        using Sec = std::chrono::seconds;
        const path older = env.create_file("older_file", 1);

        std::this_thread::sleep_for(Sec(2));
        const path from = env.create_file("update_from", 55);

        std::this_thread::sleep_for(Sec(2));
        const path newer = env.create_file("newer_file", 2);

        std::error_code ec;
        TEST_REQUIRE(fs::copy_file(from, older, copy_options::update_existing, ec) == true);
        TEST_CHECK(!ec);
        TEST_CHECK(file_size(older) == 55);

        TEST_REQUIRE(fs::copy_file(from, newer, copy_options::update_existing, ec) == false);
        TEST_CHECK(!ec);
        TEST_CHECK(file_size(newer) == 2);
    }
    { // skip_existing
        const path file2 = env.create_file("file2", 55);
        std::error_code ec;
        TEST_REQUIRE(fs::copy_file(file, file2, copy_options::skip_existing, ec) == false);
        TEST_CHECK(!ec);
        TEST_CHECK(file_size(file2) == 55);
    }
}

TEST_CASE(test_attributes_get_copied)
{
    scoped_test_env env;
    const path file = env.create_file("file1", 42);
    const path dest = env.make_env_path("file2");
    auto st = status(file);
    perms default_perms = st.permissions();
    perms new_perms = perms::owner_read;
    permissions(file, new_perms);
    std::error_code ec;
    TEST_REQUIRE(fs::copy_file(file, dest, ec) == true);
    TEST_CHECK(!ec);
    auto new_st = status(dest);
    TEST_CHECK(new_st.permissions() == new_perms);
}

TEST_CASE(copy_dir_test)
{
    scoped_test_env env;
    const path file = env.create_file("file1", 42);
    const path dest = env.create_dir("dir1");
    std::error_code ec;
    TEST_CHECK(fs::copy_file(file, dest, ec) == false);
    TEST_CHECK(ec);
    ec.clear();
    TEST_CHECK(fs::copy_file(dest, file, ec) == false);
    TEST_CHECK(ec);
}

TEST_SUITE_END()
