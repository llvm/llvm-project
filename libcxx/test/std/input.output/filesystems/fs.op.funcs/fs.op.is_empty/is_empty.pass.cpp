//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// bool is_empty(path const& p);
// bool is_empty(path const& p, std::error_code& ec);

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "assert_macros.h"
#include "test_macros.h"
#include "filesystem_test_helper.h"

using namespace fs;

static void signature_test()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(is_empty(p, ec));
    ASSERT_NOT_NOEXCEPT(is_empty(p));
}

static void test_exist_not_found()
{
    static_test_env static_env;
    const path p = static_env.DNE;
    std::error_code ec;
    assert(is_empty(p, ec) == false);
    assert(ec);
    TEST_THROWS_TYPE(filesystem_error, is_empty(p));
}

static void test_is_empty_directory()
{
    static_test_env static_env;
    assert(!is_empty(static_env.Dir));
    assert(!is_empty(static_env.SymlinkToDir));
}

static void test_is_empty_directory_dynamic()
{
    scoped_test_env env;
    assert(is_empty(env.test_root));
    env.create_file("foo", 42);
    assert(!is_empty(env.test_root));
}

static void test_is_empty_file()
{
    static_test_env static_env;
    assert(is_empty(static_env.EmptyFile));
    assert(!is_empty(static_env.NonEmptyFile));
}

static void test_is_empty_fails()
{
    scoped_test_env env;
#ifdef _WIN32
    // Windows doesn't support setting perms::none to trigger failures
    // reading directories; test using a special inaccessible directory
    // instead.
    const path p = GetWindowsInaccessibleDir();
    if (p.empty())
        return;
#else
    const path dir = env.create_dir("dir");
    const path p = env.create_dir("dir/dir2");
    permissions(dir, perms::none);
#endif

    std::error_code ec;
    assert(is_empty(p, ec) == false);
    assert(ec);

    TEST_THROWS_TYPE(filesystem_error, is_empty(p));
}

static void test_directory_access_denied()
{
    scoped_test_env env;
#ifdef _WIN32
    // Windows doesn't support setting perms::none to trigger failures
    // reading directories; test using a special inaccessible directory
    // instead.
    const path dir = GetWindowsInaccessibleDir();
    if (dir.empty())
        return;
#else
    const path dir = env.create_dir("dir");
    const path file1 = env.create_file("dir/file", 42);
    permissions(dir, perms::none);
#endif

    std::error_code ec = GetTestEC();
    assert(is_empty(dir, ec) == false);
    assert(ec);
    assert(ec != GetTestEC());

    TEST_THROWS_TYPE(filesystem_error, is_empty(dir));
}


#ifndef _WIN32
static void test_fifo_fails()
{
    scoped_test_env env;
    const path fifo = env.create_fifo("fifo");

    std::error_code ec = GetTestEC();
    assert(is_empty(fifo, ec) == false);
    assert(ec);
    assert(ec != GetTestEC());

    TEST_THROWS_TYPE(filesystem_error, is_empty(fifo));
}
#endif // _WIN32

int main(int, char**) {
    signature_test();
    test_exist_not_found();
    test_is_empty_directory();
    test_is_empty_directory_dynamic();
    test_is_empty_file();
    test_is_empty_fails();
    test_directory_access_denied();
#ifndef _WIN32
    test_fifo_fails();
#endif

    return 0;
}
