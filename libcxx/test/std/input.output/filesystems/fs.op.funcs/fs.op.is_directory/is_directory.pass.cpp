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

// <filesystem>

// bool is_directory(file_status s) noexcept
// bool is_directory(path const& p);
// bool is_directory(path const& p, std::error_code& ec) noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "assert_macros.h"
#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void signature_test()
{
    file_status s; ((void)s);
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOEXCEPT(is_directory(s));
    ASSERT_NOEXCEPT(is_directory(p, ec));
    ASSERT_NOT_NOEXCEPT(is_directory(p));
}

static void is_directory_status_test()
{
    struct TestCase {
        file_type type;
        bool expect;
    };
    const TestCase testCases[] = {
        {file_type::none, false},
        {file_type::not_found, false},
        {file_type::regular, false},
        {file_type::directory, true},
        {file_type::symlink, false},
        {file_type::block, false},
        {file_type::character, false},
        {file_type::fifo, false},
        {file_type::socket, false},
        {file_type::unknown, false}
    };
    for (auto& TC : testCases) {
        file_status s(TC.type);
        assert(is_directory(s) == TC.expect);
    }
}

static void test_exist_not_found()
{
    static_test_env static_env;
    const path p = static_env.DNE;
    assert(is_directory(p) == false);
}

static void static_env_test()
{
    static_test_env static_env;
    assert(is_directory(static_env.Dir));
    assert(is_directory(static_env.SymlinkToDir));
    assert(!is_directory(static_env.File));
}

static void test_is_directory_fails()
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
    assert(is_directory(p, ec) == false);
    assert(ec);

    TEST_THROWS_TYPE(filesystem_error, is_directory(p));
}

int main(int, char**) {
    signature_test();
    is_directory_status_test();
    test_exist_not_found();
    static_env_test();
    test_is_directory_fails();

    return 0;
}
