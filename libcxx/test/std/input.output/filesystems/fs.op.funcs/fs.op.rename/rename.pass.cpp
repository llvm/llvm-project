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

// void rename(const path& old_p, const path& new_p);
// void rename(const path& old_p,  const path& new_p, error_code& ec) noexcept;

#include <filesystem>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void test_signatures()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(fs::rename(p, p)), void);
    ASSERT_SAME_TYPE(decltype(fs::rename(p, p, ec)), void);

    ASSERT_NOT_NOEXCEPT(fs::rename(p, p));
    ASSERT_NOEXCEPT(fs::rename(p, p, ec));
}

static void test_error_reporting()
{
    auto checkThrow = [](path const& f, path const& t, const std::error_code& ec)
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            fs::rename(f, t);
            return false;
        } catch (filesystem_error const& err) {
            return err.path1() == f
                && err.path2() == t
                && err.code() == ec;
        }
#else
        ((void)f); ((void)t); ((void)ec);
        return true;
#endif
    };
    scoped_test_env env;
    const path dne = env.make_env_path("dne");
    const path file = env.create_file("file1", 42);
    const path dir = env.create_dir("dir1");
    struct TestCase {
      path from;
      path to;
    } cases[] = {
        {dne, dne},
        {file, dir},
#ifndef _WIN32
        // The spec doesn't say that this case must be an error; fs.op.rename
        // note 1.2.1 says that a file may be overwritten by a rename.
        // On Windows, with rename() implemented with MoveFileExW, overwriting
        // a file with a directory is not an error.
        {dir, file},
#endif
    };
    for (auto& TC : cases) {
        auto from_before = status(TC.from);
        auto to_before = status(TC.to);
        std::error_code ec;
        rename(TC.from, TC.to, ec);
        assert(ec);
        assert(from_before.type() == status(TC.from).type());
        assert(to_before.type() == status(TC.to).type());
        assert(checkThrow(TC.from, TC.to, ec));
    }
}

static void basic_rename_test()
{
    scoped_test_env env;

    const std::error_code set_ec = std::make_error_code(std::errc::address_in_use);
    const path file = env.create_file("file1", 42);
    { // same file
        std::error_code ec = set_ec;
        rename(file, file, ec);
        assert(!ec);
        assert(is_regular_file(file));
        assert(file_size(file) == 42);
    }
    const path sym = env.create_symlink(file, "sym");
    { // file -> symlink
        std::error_code ec = set_ec;
        rename(file, sym, ec);
        assert(!ec);
        assert(!exists(file));
        assert(is_regular_file(symlink_status(sym)));
        assert(file_size(sym) == 42);
    }
    const path file2 = env.create_file("file2", 42);
    const path file3 = env.create_file("file3", 100);
    { // file -> file
        std::error_code ec = set_ec;
        rename(file2, file3, ec);
        assert(!ec);
        assert(!exists(file2));
        assert(is_regular_file(file3));
        assert(file_size(file3) == 42);
    }
    const path dne = env.make_env_path("dne");
    const path bad_sym = env.create_symlink(dne, "bad_sym");
    const path bad_sym_dest = env.make_env_path("bad_sym2");
    { // bad-symlink
        std::error_code ec = set_ec;
        rename(bad_sym, bad_sym_dest, ec);
        assert(!ec);
        assert(!exists(symlink_status(bad_sym)));
        assert(is_symlink(bad_sym_dest));
        assert(read_symlink(bad_sym_dest) == dne);
    }
}

static void basic_rename_dir_test()
{
    static_test_env env;
    const std::error_code set_ec = std::make_error_code(std::errc::address_in_use);
    const path new_dir = env.makePath("new_dir");
    { // dir -> dir (with contents)
        std::error_code ec = set_ec;
        rename(env.Dir, new_dir, ec);
        assert(!ec);
        assert(!exists(env.Dir));
        assert(is_directory(new_dir));
        assert(exists(new_dir / "file1"));
    }
#ifdef _WIN32
    // On Windows, renaming a directory over a file isn't an error (this
    // case is skipped in test_error_reporting above).
    { // dir -> file
        std::error_code ec = set_ec;
        rename(new_dir, env.NonEmptyFile, ec);
        assert(!ec);
        assert(!exists(new_dir));
        assert(is_directory(env.NonEmptyFile));
        assert(exists(env.NonEmptyFile / "file1"));
    }
#endif
}

int main(int, char**) {
    test_signatures();
    test_error_reporting();
    basic_rename_test();
    basic_rename_dir_test();

    return 0;
}
