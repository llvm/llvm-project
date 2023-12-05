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

// bool remove(const path& p);
// bool remove(const path& p, error_code& ec) noexcept;

#include <filesystem>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void test_signatures()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(fs::remove(p)), bool);
    ASSERT_SAME_TYPE(decltype(fs::remove(p, ec)), bool);

    ASSERT_NOT_NOEXCEPT(fs::remove(p));
    ASSERT_NOEXCEPT(fs::remove(p, ec));
}

static void test_error_reporting()
{
    auto checkThrow = [](path const& f, const std::error_code& ec)
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            fs::remove(f);
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
    scoped_test_env env;
    const path non_empty_dir = env.create_dir("dir");
    env.create_file(non_empty_dir / "file1", 42);
    const path bad_perms_dir = env.create_dir("bad_dir");
    const path file_in_bad_dir = env.create_file(bad_perms_dir / "file", 42);
    permissions(bad_perms_dir, perms::none);
    const path testCases[] = {
        non_empty_dir,
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
        // Windows doesn't support setting perms::none on a directory to
        // stop it from being accessed. And a fictional file under
        // GetWindowsInaccessibleDir() doesn't cause fs::remove() to report
        // errors, it just returns false cleanly.
        file_in_bad_dir,
#endif
    };
    for (auto& p : testCases) {
        std::error_code ec;

        assert(!fs::remove(p, ec));
        assert(ec);
        assert(checkThrow(p, ec));
    }

    // PR#35780
    const path testCasesNonexistant[] = {
        "",
        env.make_env_path("dne")
    };

    for (auto& p : testCasesNonexistant) {
        std::error_code ec;

        assert(!fs::remove(p, ec));
        assert(!ec);
    }
}

static void basic_remove_test()
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

int main(int, char**) {
    test_signatures();
    test_error_reporting();
    basic_remove_test();
    return 0;
}
