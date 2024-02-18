//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// void copy_symlink(const path& existing_symlink, const path& new_symlink);
// void copy_symlink(const path& existing_symlink, const path& new_symlink,
//                   error_code& ec) noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void test_signatures()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(fs::copy_symlink(p, p));
    ASSERT_NOEXCEPT(fs::copy_symlink(p, p, ec));
}


static void test_error_reporting()
{
    auto checkThrow = [](path const& f, path const& t, const std::error_code& ec)
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            fs::copy_symlink(f, t);
            return true;
        } catch (filesystem_error const& err) {
            return err.path1() == f
                && err.code() == ec;
        }
#else
        ((void)f); ((void)t); ((void)ec);
        return true;
#endif
    };

    scoped_test_env env;
    const path file = env.create_file("file1", 42);
    const path file2 = env.create_file("file2", 55);
    const path sym = env.create_symlink(file, "sym");
    const path dir = env.create_dir("dir");
    const path dne = env.make_env_path("dne");
    { // from is a file, not a symlink
        std::error_code ec;
        fs::copy_symlink(file, dne, ec);
        assert(ec);
        assert(checkThrow(file, dne, ec));
    }
    { // from is a file, not a symlink
        std::error_code ec;
        fs::copy_symlink(dir, dne, ec);
        assert(ec);
        assert(checkThrow(dir, dne, ec));
    }
    { // destination exists
        std::error_code ec;
        fs::copy_symlink(sym, file2, ec);
        assert(ec);
    }
}

static void copy_symlink_basic()
{
    scoped_test_env env;
    const path dir = env.create_dir("dir");
    const path dir_sym = env.create_directory_symlink(dir, "dir_sym");
    const path file = env.create_file("file", 42);
    const path file_sym = env.create_symlink(file, "file_sym");
    { // test for directory symlinks
        const path dest = env.make_env_path("dest1");
        std::error_code ec;
        fs::copy_symlink(dir_sym, dest, ec);
        assert(!ec);
        assert(is_symlink(dest));
        assert(equivalent(dest, dir));
    }
    { // test for file symlinks
        const path dest = env.make_env_path("dest2");
        std::error_code ec;
        fs::copy_symlink(file_sym, dest, ec);
        assert(!ec);
        assert(is_symlink(dest));
        assert(equivalent(dest, file));
    }
}

int main(int, char**) {
    test_signatures();
    test_error_reporting();
    copy_symlink_basic();

    return 0;
}
