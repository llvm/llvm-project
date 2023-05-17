//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// path read_symlink(const path& p);
// path read_symlink(const path& p, error_code& ec);

#include "filesystem_include.h"

#include "test_macros.h"
#include "filesystem_test_helper.h"

using namespace fs;

static void test_signatures()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(fs::read_symlink(p)), fs::path);
    ASSERT_SAME_TYPE(decltype(fs::read_symlink(p, ec)), fs::path);

    ASSERT_NOT_NOEXCEPT(fs::read_symlink(p));
    // Not noexcept because of narrow contract
    ASSERT_NOT_NOEXCEPT(fs::read_symlink(p, ec));
}

static void test_error_reporting()
{
    auto checkThrow = [](path const& f, const std::error_code& ec)
    {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            (void)fs::read_symlink(f);
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
    const path cases[] = {
        env.make_env_path("dne"),
        env.create_file("file", 42),
        env.create_dir("dir")
    };
    for (path const& p : cases) {
        std::error_code ec;
        const path ret = fs::read_symlink(p, ec);
        assert(ec);
        assert(ret == path{});
        assert(checkThrow(p, ec));
    }

}

static void basic_symlink_test()
{
    scoped_test_env env;
    const path dne = env.make_env_path("dne");
    const path file = env.create_file("file", 42);
    const path dir = env.create_dir("dir");
    const path link = env.create_symlink(dne, "link");
    const path nested_link = env.make_env_path("nested_link");
    create_symlink(link, nested_link);
    struct TestCase {
      path symlink;
      path expected;
    } testCases[] = {
        {env.create_symlink(dne, "dne_link"), dne},
        {env.create_symlink(file, "file_link"), file},
        {env.create_directory_symlink(dir, "dir_link"), dir},
        {nested_link, link}
    };
    for (auto& TC : testCases) {
        std::error_code ec = std::make_error_code(std::errc::address_in_use);
        const path ret = read_symlink(TC.symlink, ec);
        assert(!ec);
        assert(ret == TC.expected);
    }
}

int main(int, char**) {
    test_signatures();
    test_error_reporting();
    basic_symlink_test();

    return 0;
}
