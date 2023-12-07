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

// void copy(const path& from, const path& to);
// void copy(const path& from, const path& to, error_code& ec);
// void copy(const path& from, const path& to, copy_options options);
// void copy(const path& from, const path& to, copy_options options,
//           error_code& ec);

#include "filesystem_include.h"
#include <type_traits>
#include <cstddef>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"

using namespace fs;

using CO = fs::copy_options;

static void signature_test()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    const copy_options opts{}; ((void)opts);
    ASSERT_NOT_NOEXCEPT(fs::copy(p, p));
    ASSERT_NOT_NOEXCEPT(fs::copy(p, p, ec));
    ASSERT_NOT_NOEXCEPT(copy(p, p, opts));
    ASSERT_NOT_NOEXCEPT(copy(p, p, opts, ec));
}

// There are 4 cases is the proposal for absolute path.
// Each scope tests one of the cases.
static void test_error_reporting()
{
    auto checkThrow = [](path const& f, path const& t, const std::error_code& ec)
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            fs::copy(f, t);
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

    static_test_env static_env;
    scoped_test_env env;
    const path file = env.create_file("file1", 42);
    const path dir = env.create_dir("dir");
#ifndef _WIN32
    const path fifo = env.create_fifo("fifo");
    assert(is_other(fifo));
#endif

    const auto test_ec = GetTestEC();

    // !exists(f)
    {
        std::error_code ec = test_ec;
        const path f = static_env.DNE;
        const path t = env.test_root;
        fs::copy(f, t, ec);
        assert(ec);
        assert(ec != test_ec);
        assert(checkThrow(f, t, ec));
    }
    { // equivalent(f, t) == true
        std::error_code ec = test_ec;
        fs::copy(file, file, ec);
        assert(ec);
        assert(ec != test_ec);
        assert(checkThrow(file, file, ec));
    }
    { // is_directory(from) && is_file(to)
        std::error_code ec = test_ec;
        fs::copy(dir, file, ec);
        assert(ec);
        assert(ec != test_ec);
        assert(checkThrow(dir, file, ec));
    }
#ifndef _WIN32
    { // is_other(from)
        std::error_code ec = test_ec;
        fs::copy(fifo, dir, ec);
        assert(ec);
        assert(ec != test_ec);
        assert(checkThrow(fifo, dir, ec));
    }
    { // is_other(to)
        std::error_code ec = test_ec;
        fs::copy(file, fifo, ec);
        assert(ec);
        assert(ec != test_ec);
        assert(checkThrow(file, fifo, ec));
    }
#endif
}

static void from_is_symlink()
{
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    const path symlink = env.create_symlink(file, "sym");
    const path dne = env.make_env_path("dne");

    { // skip symlinks
        std::error_code ec = GetTestEC();
        fs::copy(symlink, dne, copy_options::skip_symlinks, ec);
        assert(!ec);
        assert(!exists(dne));
    }
    {
        const path dest = env.make_env_path("dest");
        std::error_code ec = GetTestEC();
        fs::copy(symlink, dest, copy_options::copy_symlinks, ec);
        assert(!ec);
        assert(exists(dest));
        assert(is_symlink(dest));
    }
    { // copy symlink but target exists
        std::error_code ec = GetTestEC();
        fs::copy(symlink, file, copy_options::copy_symlinks, ec);
        assert(ec);
        assert(ec != GetTestEC());
    }
    { // create symlinks but target exists
        std::error_code ec = GetTestEC();
        fs::copy(symlink, file, copy_options::create_symlinks, ec);
        assert(ec);
        assert(ec != GetTestEC());
    }
}

static void from_is_regular_file()
{
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    const path dir = env.create_dir("dir");
    { // skip copy because of directory
        const path dest = env.make_env_path("dest1");
        std::error_code ec = GetTestEC();
        fs::copy(file, dest, CO::directories_only, ec);
        assert(!ec);
        assert(!exists(dest));
    }
    { // create symlink to file
        const path dest = env.make_env_path("sym");
        std::error_code ec = GetTestEC();
        fs::copy(file, dest, CO::create_symlinks, ec);
        assert(!ec);
        assert(is_symlink(dest));
        assert(equivalent(file, canonical(dest)));
    }
    { // create hard link to file
        const path dest = env.make_env_path("hardlink");
        assert(hard_link_count(file) == 1);
        std::error_code ec = GetTestEC();
        fs::copy(file, dest, CO::create_hard_links, ec);
        assert(!ec);
        assert(exists(dest));
        assert(hard_link_count(file) == 2);
    }
    { // is_directory(t)
        const path dest_dir = env.create_dir("dest_dir");
        const path expect_dest = dest_dir / file.filename();
        std::error_code ec = GetTestEC();
        fs::copy(file, dest_dir, ec);
        assert(!ec);
        assert(is_regular_file(expect_dest));
    }
    { // otherwise copy_file(from, to, ...)
        const path dest = env.make_env_path("file_copy");
        std::error_code ec = GetTestEC();
        fs::copy(file, dest, ec);
        assert(!ec);
        assert(is_regular_file(dest));
    }
}

static void from_is_directory()
{
    struct FileInfo {
        path filename;
        std::size_t size;
    };
    const FileInfo files[] = {
        {"file1", 0},
        {"file2", 42},
        {"file3", 300}
    };
    scoped_test_env env;
    const path dir = env.create_dir("dir");
    const path nested_dir_name = "dir2";
    const path nested_dir = env.create_dir("dir/dir2");

    for (auto& FI : files) {
        env.create_file(dir / FI.filename, FI.size);
        env.create_file(nested_dir / FI.filename, FI.size);
    }
    { // test for non-existent directory
        const path dest = env.make_env_path("dest_dir1");
        std::error_code ec = GetTestEC();
        fs::copy(dir, dest, ec);
        assert(!ec);
        assert(is_directory(dest));
        for (auto& FI : files) {
            path created = dest / FI.filename;
            assert(is_regular_file(created));
            assert(file_size(created) == FI.size);
        }
        assert(!is_directory(dest / nested_dir_name));
    }
    { // test for existing directory
        const path dest = env.create_dir("dest_dir2");
        std::error_code ec = GetTestEC();
        fs::copy(dir, dest, ec);
        assert(!ec);
        assert(is_directory(dest));
        for (auto& FI : files) {
            path created = dest / FI.filename;
            assert(is_regular_file(created));
            assert(file_size(created) == FI.size);
        }
        assert(!is_directory(dest / nested_dir_name));
    }
    { // test recursive copy
        const path dest = env.make_env_path("dest_dir3");
        std::error_code ec = GetTestEC();
        fs::copy(dir, dest, CO::recursive, ec);
        assert(!ec);
        assert(is_directory(dest));
        const path nested_dest = dest / nested_dir_name;
        assert(is_directory(nested_dest));
        for (auto& FI : files) {
            path created = dest / FI.filename;
            path nested_created = nested_dest / FI.filename;
            assert(is_regular_file(created));
            assert(file_size(created) == FI.size);
            assert(is_regular_file(nested_created));
            assert(file_size(nested_created) == FI.size);
        }
    }
}

static void test_copy_symlinks_to_symlink_dir()
{
    scoped_test_env env;
    const path file1 = env.create_file("file1", 42);
    const path file2 = env.create_file("file2", 101);
    const path file2_sym = env.create_symlink(file2, "file2_sym");
    const path dir = env.create_dir("dir");
    const path dir_sym = env.create_directory_symlink(dir, "dir_sym");
    {
        std::error_code ec = GetTestEC();
        fs::copy(file1, dir_sym, copy_options::copy_symlinks, ec);
        assert(!ec);
        const path dest = env.make_env_path("dir/file1");
        assert(exists(dest));
        assert(!is_symlink(dest));
        assert(file_size(dest) == 42);
    }
}


static void test_dir_create_symlink()
{
    scoped_test_env env;
    const path dir = env.create_dir("dir1");
    const path dest = env.make_env_path("dne");
    {
        std::error_code ec = GetTestEC();
        fs::copy(dir, dest, copy_options::create_symlinks, ec);
        assert(ErrorIs(ec, std::errc::is_a_directory));
        assert(!exists(dest));
        assert(!is_symlink(dest));
    }
    {
        std::error_code ec = GetTestEC();
        fs::copy(dir, dest, copy_options::create_symlinks|copy_options::recursive, ec);
        assert(ErrorIs(ec, std::errc::is_a_directory));
        assert(!exists(dest));
        assert(!is_symlink(dest));
    }
}

static void test_otherwise_no_effects_clause()
{
    scoped_test_env env;
    const path dir = env.create_dir("dir1");
    { // skip copy because of directory
        const path dest = env.make_env_path("dest1");
        std::error_code ec;
        fs::copy(dir, dest, CO::directories_only, ec);
        assert(!ec);
        assert(!exists(dest));
    }
}

int main(int, char**) {
    signature_test();
    test_error_reporting();
    from_is_symlink();
    from_is_regular_file();
    from_is_directory();
    test_copy_symlinks_to_symlink_dir();
    test_dir_create_symlink();
    test_otherwise_no_effects_clause();

    return 0;
}
