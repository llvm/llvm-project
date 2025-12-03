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

// void resize_file(const path& p, uintmax_t new_size);
// void resize_file(const path& p, uintmax_t new_size, error_code& ec) noexcept;

#include <filesystem>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void test_signatures()
{
    const path p; ((void)p);
    std::uintmax_t i; ((void)i);
    std::error_code ec; ((void)ec);

    ASSERT_SAME_TYPE(decltype(fs::resize_file(p, i)), void);
    ASSERT_SAME_TYPE(decltype(fs::resize_file(p, i, ec)), void);

    ASSERT_NOT_NOEXCEPT(fs::resize_file(p, i));
    ASSERT_NOEXCEPT(fs::resize_file(p, i, ec));
}

static void test_error_reporting()
{
    auto checkThrow = [](path const& f, std::uintmax_t s, const std::error_code& ec)
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            fs::resize_file(f, s);
            return false;
        } catch (filesystem_error const& err) {
            return err.path1() == f
                && err.path2() == ""
                && err.code() == ec;
        }
#else
        ((void)f); ((void)s); ((void)ec);
        return true;
#endif
    };
    scoped_test_env env;
    const path dne = env.make_env_path("dne");
    const path bad_sym = env.create_symlink(dne, "sym");
    const path dir = env.create_dir("dir1");
    const path cases[] = {
        dne, bad_sym, dir
    };
    for (auto& p : cases) {
        std::error_code ec;
        resize_file(p, 42, ec);
        assert(ec);
        assert(checkThrow(p, 42, ec));
    }
}

static void basic_resize_file_test()
{
    scoped_test_env env;
    const path file1 = env.create_file("file1", 42);
    const auto set_ec = std::make_error_code(std::errc::address_in_use);
    { // grow file
        const std::uintmax_t new_s = 100;
        std::error_code ec = set_ec;
        resize_file(file1, new_s, ec);
        assert(!ec);
        assert(file_size(file1) == new_s);
    }
    { // shrink file
        const std::uintmax_t new_s = 1;
        std::error_code ec = set_ec;
        resize_file(file1, new_s, ec);
        assert(!ec);
        assert(file_size(file1) == new_s);
    }
    { // shrink file to zero
        const std::uintmax_t new_s = 0;
        std::error_code ec = set_ec;
        resize_file(file1, new_s, ec);
        assert(!ec);
        assert(file_size(file1) == new_s);
    }
    const path sym = env.create_symlink(file1, "sym");
    { // grow file via symlink
        const std::uintmax_t new_s = 1024;
        std::error_code ec = set_ec;
        resize_file(sym, new_s, ec);
        assert(!ec);
        assert(file_size(file1) == new_s);
    }
}

int main(int, char**) {
    test_signatures();
    test_error_reporting();
    basic_resize_file_test();
    return 0;
}
