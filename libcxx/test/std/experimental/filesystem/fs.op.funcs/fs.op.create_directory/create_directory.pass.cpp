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

// bool create_directory(const path& p);
// bool create_directory(const path& p, error_code& ec) noexcept;
// bool create_directory(const path& p, const path& attr);
// bool create_directory(const path& p, const path& attr, error_code& ec) noexcept;

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

TEST_SUITE(filesystem_create_directory_test_suite)

TEST_CASE(test_signatures)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p)), bool);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p, ec)), bool);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p, p)), bool);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p, p, ec)), bool);
    ASSERT_NOT_NOEXCEPT(fs::create_directory(p));
    ASSERT_NOEXCEPT(fs::create_directory(p, ec));
    ASSERT_NOT_NOEXCEPT(fs::create_directory(p, p));
    ASSERT_NOEXCEPT(fs::create_directory(p, p, ec));
}


TEST_CASE(create_existing_directory)
{
    scoped_test_env env;
    const path dir = env.create_dir("dir1");
    std::error_code ec;
    TEST_CHECK(fs::create_directory(dir, ec) == false);
    TEST_CHECK(!ec);
    TEST_CHECK(is_directory(dir));
    // Test throwing version
    TEST_CHECK(fs::create_directory(dir) == false);
}

TEST_CASE(create_directory_one_level)
{
    scoped_test_env env;
    const path dir = env.make_env_path("dir1");
    std::error_code ec;
    TEST_CHECK(fs::create_directory(dir, ec) == true);
    TEST_CHECK(!ec);
    TEST_CHECK(is_directory(dir));

    auto st = status(dir);
    perms owner_perms = perms::owner_all;
    perms gperms = perms::group_all;
    perms other_perms = perms::others_read | perms::others_exec;
#if defined(__APPLE__) || defined(__FreeBSD__)
    gperms = perms::group_read | perms::group_exec;
#endif
    TEST_CHECK((st.permissions() & perms::owner_all) == owner_perms);
    TEST_CHECK((st.permissions() & perms::group_all) == gperms);
    TEST_CHECK((st.permissions() & perms::others_all) == other_perms);
}

TEST_CASE(create_directory_multi_level)
{
    scoped_test_env env;
    const path dir = env.make_env_path("dir1/dir2");
    const path dir1 = env.make_env_path("dir1");
    std::error_code ec;
    TEST_CHECK(fs::create_directory(dir, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(!is_directory(dir));
    TEST_CHECK(!is_directory(dir1));
}

TEST_CASE(dest_is_file)
{
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    std::error_code ec;
    TEST_CHECK(fs::create_directory(file, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(is_regular_file(file));
}

TEST_SUITE_END()
