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

// void create_symlink(const path& existing_symlink, const path& new_symlink);
// void create_symlink(const path& existing_symlink, const path& new_symlink,
//                   error_code& ec) noexcept;

#include <filesystem>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void test_signatures()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(fs::create_symlink(p, p));
    ASSERT_NOEXCEPT(fs::create_symlink(p, p, ec));
}

static void test_error_reporting()
{
    scoped_test_env env;
    const path file = env.create_file("file1", 42);
    const path file2 = env.create_file("file2", 55);
    const path sym = env.create_symlink(file, "sym");
    { // destination exists
        std::error_code ec;
        fs::create_symlink(sym, file2, ec);
        assert(ec);
    }
}

static void create_symlink_basic()
{
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    const path file_sym = env.create_symlink(file, "file_sym");
    const path dir = env.create_dir("dir");
    const path dir_sym = env.create_directory_symlink(dir, "dir_sym");
    {
        const path dest = env.make_env_path("dest1");
        std::error_code ec;
        fs::create_symlink(file_sym, dest, ec);
        assert(!ec);
        assert(is_symlink(dest));
        assert(equivalent(dest, file));
    }
    {
        const path dest = env.make_env_path("dest2");
        std::error_code ec;
        fs::create_directory_symlink(dir_sym, dest, ec);
        assert(!ec);
        assert(is_symlink(dest));
        assert(equivalent(dest, dir));
    }
}

static void create_symlink_dest_cleanup()
{
    scoped_test_env env;
    const path dir = env.create_dir("dir");
    const path file = env.create_file("file", 42);
    const path sym = dir / "link";
    // The target path has to be normalized to backslashes before creating
    // the link on windows, otherwise the link isn't dereferencable.
    const path sym_target = "../file";
    path sym_target_normalized = sym_target;
    sym_target_normalized.make_preferred();
    std::error_code ec;
    fs::create_symlink(sym_target, sym, ec);
    assert(!ec);
    assert(equivalent(sym, file, ec));
    const path ret = fs::read_symlink(sym, ec);
    assert(!ec);
    assert(ret.native() == sym_target_normalized.native());
}

int main(int, char**) {
    test_signatures();
    test_error_reporting();
    create_symlink_basic();
    create_symlink_dest_cleanup();

    return 0;
}
