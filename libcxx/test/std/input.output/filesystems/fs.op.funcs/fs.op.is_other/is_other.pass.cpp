//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// bool is_other(file_status s) noexcept
// bool is_other(path const& p);
// bool is_other(path const& p, std::error_code& ec) noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "assert_macros.h"
#include "test_macros.h"
#include "filesystem_test_helper.h"

using namespace fs;

static void signature_test()
{
    file_status s; ((void)s);
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOEXCEPT(is_other(s));
    ASSERT_NOEXCEPT(is_other(p, ec));
    ASSERT_NOT_NOEXCEPT(is_other(p));
}

static void is_other_status_test()
{
    struct TestCase {
        file_type type;
        bool expect;
    };
    const TestCase testCases[] = {
        {file_type::none, false},
        {file_type::not_found, false},
        {file_type::regular, false},
        {file_type::directory, false},
        {file_type::symlink, false},
        {file_type::block, true},
        {file_type::character, true},
        {file_type::fifo, true},
        {file_type::socket, true},
        {file_type::unknown, true}
    };
    for (auto& TC : testCases) {
        file_status s(TC.type);
        assert(is_other(s) == TC.expect);
    }
}

static void test_exist_not_found()
{
    static_test_env static_env;
    const path p = static_env.DNE;
    assert(is_other(p) == false);
}

static void test_is_other_fails()
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
    const path p = env.create_file("dir/file", 42);
    permissions(dir, perms::none);
#endif

    std::error_code ec;
    assert(is_other(p, ec) == false);
    assert(ec);

    TEST_THROWS_TYPE(filesystem_error, is_other(p));
}

int main(int, char**) {
    signature_test();
    is_other_status_test();
    test_exist_not_found();
    test_is_other_fails();

    return 0;
}
