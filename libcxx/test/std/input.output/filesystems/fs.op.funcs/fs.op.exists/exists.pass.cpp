//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// bool exists(file_status s) noexcept
// bool exists(path const& p);
// bool exists(path const& p, std::error_code& ec) noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(exists_test_suite)

TEST_CASE(signature_test)
{
    file_status s; ((void)s);
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOEXCEPT(exists(s));
    ASSERT_NOEXCEPT(exists(p, ec));
    ASSERT_NOT_NOEXCEPT(exists(p));
}

TEST_CASE(exists_status_test)
{
    struct TestCase {
        file_type type;
        bool expect;
    };
    const TestCase testCases[] = {
        {file_type::none, false},
        {file_type::not_found, false},
        {file_type::regular, true},
        {file_type::directory, true},
        {file_type::symlink, true},
        {file_type::block, true},
        {file_type::character, true},
        {file_type::fifo, true},
        {file_type::socket, true},
        {file_type::unknown, true}
    };
    for (auto& TC : testCases) {
        file_status s(TC.type);
        TEST_CHECK(exists(s) == TC.expect);
    }
}

TEST_CASE(test_exist_not_found)
{
    static_test_env static_env;
    const path p = static_env.DNE;
    TEST_CHECK(exists(p) == false);

    std::error_code ec = GetTestEC();
    TEST_CHECK(exists(p, ec) == false);
    TEST_CHECK(!ec);
}

TEST_CASE(test_exists_fails)
{
#ifdef _WIN32
    // Windows doesn't support setting perms::none to trigger failures
    // reading directories; test using a special inaccessible directory
    // instead.
    const path p = GetWindowsInaccessibleDir();
    TEST_REQUIRE(!p.empty());
#else
    scoped_test_env env;
    const path dir = env.create_dir("dir");
    const path p = env.create_file("dir/file", 42);
    permissions(dir, perms::none);
#endif

    std::error_code ec;
    TEST_CHECK(exists(p, ec) == false);
    TEST_CHECK(ec);

    TEST_CHECK_THROW(filesystem_error, exists(p));
}

#ifndef _WIN32
// Checking for the existence of an invalid long path name doesn't
// trigger errors on windows.
TEST_CASE(test_name_too_long) {
    std::string long_name(2500, 'a');
    const path file(long_name);

    std::error_code ec;
    TEST_CHECK(exists(file, ec) == false);
    TEST_CHECK(ec);
}
#endif

TEST_SUITE_END()
