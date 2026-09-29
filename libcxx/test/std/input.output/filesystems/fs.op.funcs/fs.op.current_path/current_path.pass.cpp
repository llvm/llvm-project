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

// path current_path();
// path current_path(error_code& ec);
// void current_path(path const&);
// void current_path(path const&, std::error_code& ec) noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void current_path_signature_test()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(current_path());
    ASSERT_NOT_NOEXCEPT(current_path(ec));
    ASSERT_NOT_NOEXCEPT(current_path(p));
    ASSERT_NOEXCEPT(current_path(p, ec));
}

static void current_path_test()
{
    std::error_code ec;
    const path p = current_path(ec);
    assert(!ec);
    assert(p.is_absolute());
    assert(is_directory(p));

    const path p2 = current_path();
    assert(p2 == p);
}

static void current_path_after_change_test()
{
    static_test_env static_env;
    CWDGuard guard;
    const path new_path = static_env.Dir;
    current_path(new_path);
    assert(current_path() == new_path);
}

static void current_path_is_file_test()
{
    static_test_env static_env;
    CWDGuard guard;
    const path p = static_env.File;
    std::error_code ec;
    const path old_p = current_path();
    current_path(p, ec);
    assert(ec);
    assert(old_p == current_path());
}

static void set_to_non_absolute_path()
{
    static_test_env static_env;
    CWDGuard guard;
    const path base = static_env.Dir;
    current_path(base);
    const path p = static_env.Dir2.filename();
    std::error_code ec;
    current_path(p, ec);
    assert(!ec);
    const path new_cwd = current_path();
    assert(new_cwd == static_env.Dir2);
    assert(new_cwd.is_absolute());
}

static void set_to_empty()
{
    const path p = "";
    std::error_code ec;
    const path old_p = current_path();
    current_path(p, ec);
    assert(ec);
    assert(old_p == current_path());
}

int main(int, char**) {
    current_path_signature_test();
    current_path_test();
    current_path_after_change_test();
    current_path_is_file_test();
    set_to_non_absolute_path();
    set_to_empty();

    return 0;
}
