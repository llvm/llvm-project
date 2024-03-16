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

// uintmax_t remove_all(const path& p);
// uintmax_t remove_all(const path& p, error_code& ec) noexcept;

#include <filesystem>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void test_signatures()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(fs::remove_all(p)), std::uintmax_t);
    ASSERT_SAME_TYPE(decltype(fs::remove_all(p, ec)), std::uintmax_t);

    ASSERT_NOT_NOEXCEPT(fs::remove_all(p));
    ASSERT_NOT_NOEXCEPT(fs::remove_all(p, ec));
}

static void test_error_reporting()
{
    scoped_test_env env;
    // Windows doesn't support setting perms::none to trigger failures
    // reading directories.
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
    auto checkThrow = [](path const& f, const std::error_code& ec)
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            fs::remove_all(f);
            return false;
        } catch (filesystem_error const& err) {
            return err.path1() == f
                && err.path2() == ""
                && err.code() == ec;
        }
#else
        ((void)f); ((void)ec);
        return true;
#endif
    };
    const path non_empty_dir = env.create_dir("dir");
    env.create_file(non_empty_dir / "file1", 42);
    const path bad_perms_dir = env.create_dir("bad_dir");
    const path file_in_bad_dir = env.create_file(bad_perms_dir / "file", 42);
    permissions(bad_perms_dir, perms::none);
    const path bad_perms_file = env.create_file("file2", 42);
    permissions(bad_perms_file, perms::none);

    const path testCases[] = {
        file_in_bad_dir
    };
    const auto BadRet = static_cast<std::uintmax_t>(-1);
    for (auto& p : testCases) {
        std::error_code ec;

        assert(fs::remove_all(p, ec) == BadRet);
        assert(ec);
        assert(checkThrow(p, ec));
    }
#endif

    // PR#35780
    const path testCasesNonexistant[] = {
        "",
        env.make_env_path("dne")
    };
    for (auto &p : testCasesNonexistant) {
        std::error_code ec;

        assert(fs::remove_all(p, ec) == 0);
        assert(!ec);
    }
}

static void basic_remove_all_test()
{
    scoped_test_env env;
    const path dne = env.make_env_path("dne");
    const path link = env.create_symlink(dne, "link");
    const path nested_link = env.make_env_path("nested_link");
    create_symlink(link, nested_link);
    const path testCases[] = {
        env.create_file("file", 42),
        env.create_dir("empty_dir"),
        nested_link,
        link
    };
    for (auto& p : testCases) {
        std::error_code ec = std::make_error_code(std::errc::address_in_use);
        assert(remove(p, ec));
        assert(!ec);
        assert(!exists(symlink_status(p)));
    }
}

static void symlink_to_dir()
{
    scoped_test_env env;
    const path dir = env.create_dir("dir");
    const path file = env.create_file(dir / "file", 42);
    const path link = env.create_directory_symlink(dir, "sym");

    {
        std::error_code ec = std::make_error_code(std::errc::address_in_use);
        assert(remove_all(link, ec) == 1);
        assert(!ec);
        assert(!exists(symlink_status(link)));
        assert(exists(dir));
        assert(exists(file));
    }
}


static void nested_dir()
{
    scoped_test_env env;
    const path dir = env.create_dir("dir");
    const path dir1 = env.create_dir(dir / "dir1");
    const path out_of_dir_file = env.create_file("file1", 42);
    const path all_files[] = {
        dir, dir1,
        env.create_file(dir / "file1", 42),
        env.create_symlink(out_of_dir_file, dir / "sym1"),
        env.create_file(dir1 / "file2", 42),
        env.create_directory_symlink(dir, dir1 / "sym2")
    };
    const std::size_t expected_count = sizeof(all_files) / sizeof(all_files[0]);

    std::error_code ec = std::make_error_code(std::errc::address_in_use);
    assert(remove_all(dir, ec) == expected_count);
    assert(!ec);
    for (auto const& p : all_files) {
        assert(!exists(symlink_status(p)));
    }
    assert(exists(out_of_dir_file));
}

int main(int, char**) {
    test_signatures();
    test_error_reporting();
    basic_remove_all_test();
    symlink_to_dir();
    nested_dir();

    return 0;
}
