//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// XFAIL: LIBCXX-AIX-FIXME

// <filesystem>

// bool is_empty(path const& p);
// bool is_empty(path const& p, std::error_code& ec);

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(is_empty_test_suite)

TEST_CASE(signature_test)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(is_empty(p, ec));
    ASSERT_NOT_NOEXCEPT(is_empty(p));
}

TEST_CASE(test_exist_not_found)
{
    static_test_env static_env;
    const path p = static_env.DNE;
    std::error_code ec;
    TEST_CHECK(is_empty(p, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK_THROW(filesystem_error, is_empty(p));
}

TEST_CASE(test_is_empty_directory)
{
    static_test_env static_env;
    TEST_CHECK(!is_empty(static_env.Dir));
    TEST_CHECK(!is_empty(static_env.SymlinkToDir));
}

TEST_CASE(test_is_empty_directory_dynamic)
{
    scoped_test_env env;
    TEST_CHECK(is_empty(env.test_root));
    env.create_file("foo", 42);
    TEST_CHECK(!is_empty(env.test_root));
}

TEST_CASE(test_is_empty_file)
{
    static_test_env static_env;
    TEST_CHECK(is_empty(static_env.EmptyFile));
    TEST_CHECK(!is_empty(static_env.NonEmptyFile));
}

TEST_CASE(test_is_empty_fails)
{
    scoped_test_env env;
#ifdef _WIN32
    // Windows doesn't support setting perms::none to trigger failures
    // reading directories; test using a special inaccessible directory
    // instead.
    const path p = GetWindowsInaccessibleDir();
    if (p.empty())
        TEST_UNSUPPORTED();
#else
    const path dir = env.create_dir("dir");
    const path p = env.create_dir("dir/dir2");
    permissions(dir, perms::none);
#endif

    std::error_code ec;
    TEST_CHECK(is_empty(p, ec) == false);
    TEST_CHECK(ec);

    TEST_CHECK_THROW(filesystem_error, is_empty(p));
}

TEST_CASE(test_directory_access_denied)
{
    scoped_test_env env;
#ifdef _WIN32
    // Windows doesn't support setting perms::none to trigger failures
    // reading directories; test using a special inaccessible directory
    // instead.
    const path dir = GetWindowsInaccessibleDir();
    if (dir.empty())
        TEST_UNSUPPORTED();
#else
    const path dir = env.create_dir("dir");
    const path file1 = env.create_file("dir/file", 42);
    permissions(dir, perms::none);
#endif

    std::error_code ec = GetTestEC();
    TEST_CHECK(is_empty(dir, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(ec != GetTestEC());

    TEST_CHECK_THROW(filesystem_error, is_empty(dir));
}


#ifndef _WIN32
TEST_CASE(test_fifo_fails)
{
    scoped_test_env env;
    const path fifo = env.create_fifo("fifo");

    std::error_code ec = GetTestEC();
    TEST_CHECK(is_empty(fifo, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(ec != GetTestEC());

    TEST_CHECK_THROW(filesystem_error, is_empty(fifo));
}
#endif

TEST_SUITE_END()
