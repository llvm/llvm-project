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

// path temp_directory_path();
// path temp_directory_path(error_code& ec);

#include "filesystem_include.h"
#include <memory>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"

using namespace fs;

void PutEnv(std::string var, fs::path value) {
    assert(utils::setenv(var.c_str(), value.string().c_str(), /* overwrite */ 1) == 0);
}

void UnsetEnv(std::string var) {
    assert(utils::unsetenv(var.c_str()) == 0);
}

static void signature_test()
{
    std::error_code ec; ((void)ec);
    ASSERT_NOT_NOEXCEPT(temp_directory_path());
    ASSERT_NOT_NOEXCEPT(temp_directory_path(ec));
}

static void basic_tests()
{
    scoped_test_env env;
    const path dne = env.make_env_path("dne");
    const path file = env.create_file("file", 42);
#ifdef _WIN32
    // Windows doesn't support setting perms::none to trigger failures
    // reading directories; test using a special inaccessible directory
    // instead.
    const path inaccessible_dir = GetWindowsInaccessibleDir();
#else
    const path dir_perms = env.create_dir("bad_perms_dir");
    const path inaccessible_dir = env.create_dir("bad_perms_dir/nested");
    permissions(dir_perms, perms::none);
#endif
    LIBCPP_ONLY(const std::errc expect_errc = std::errc::not_a_directory);
    struct TestCase {
      std::string name;
      path p;
    } cases[] = {
#ifdef _WIN32
        {"TMP", env.create_dir("dir1")},
        {"TEMP", env.create_dir("dir2")},
        {"USERPROFILE", env.create_dir("dir3")}
#else
        {"TMPDIR", env.create_dir("dir1")},
        {"TMP", env.create_dir("dir2")},
        {"TEMP", env.create_dir("dir3")},
        {"TEMPDIR", env.create_dir("dir4")}
#endif
    };
    TestCase ignored_cases[] = {
#ifdef _WIN32
        {"TMPDIR", env.create_dir("dir5")},
        {"TEMPDIR", env.create_dir("dir6")},
#else
        {"USERPROFILE", env.create_dir("dir5")},
#endif
    };
    for (auto& TC : cases) {
        PutEnv(TC.name, TC.p);
    }
    for (auto& TC : cases) {
        std::error_code ec = GetTestEC();
        path ret = temp_directory_path(ec);
        assert(!ec);
        assert(ret == TC.p);
        assert(is_directory(ret));

        // Set the env variable to a path that does not exist and check
        // that it fails.
        PutEnv(TC.name, dne);
        ec = GetTestEC();
        ret = temp_directory_path(ec);
        LIBCPP_ONLY(assert(ErrorIs(ec, expect_errc)));
        assert(ec != GetTestEC());
        assert(ec);
        assert(ret == "");

        // Set the env variable to point to a file and check that it fails.
        PutEnv(TC.name, file);
        ec = GetTestEC();
        ret = temp_directory_path(ec);
        LIBCPP_ONLY(assert(ErrorIs(ec, expect_errc)));
        assert(ec != GetTestEC());
        assert(ec);
        assert(ret == "");

        if (!inaccessible_dir.empty()) {
            // Set the env variable to point to a dir we can't access
            PutEnv(TC.name, inaccessible_dir);
            ec = GetTestEC();
            ret = temp_directory_path(ec);
            assert(ErrorIs(ec, std::errc::permission_denied));
            assert(ret == "");
        }

        // Set the env variable to point to a non-existent dir
        PutEnv(TC.name, TC.p / "does_not_exist");
        ec = GetTestEC();
        ret = temp_directory_path(ec);
        assert(ec != GetTestEC());
        assert(ec);
        assert(ret == "");

        // Finally erase this env variable
        UnsetEnv(TC.name);
    }
    // No env variables are defined
    path fallback;
    {
        std::error_code ec = GetTestEC();
        path ret = temp_directory_path(ec);
        assert(!ec);
#if defined(_WIN32)
        // On Windows, the function falls back to the Windows folder.
        wchar_t win_dir[MAX_PATH];
        DWORD win_dir_sz = GetWindowsDirectoryW(win_dir, MAX_PATH);
        assert(win_dir_sz > 0 && win_dir_sz < MAX_PATH);
        assert(win_dir[win_dir_sz-1] != L'\\');
        assert(ret == win_dir);
#elif defined(__ANDROID__)
        assert(ret == "/data/local/tmp");
#else
        assert(ret == "/tmp");
#endif
        assert(is_directory(ret));
        fallback = ret;
    }
    for (auto& TC : ignored_cases) {
        // Check that certain variables are ignored
        PutEnv(TC.name, TC.p);
        std::error_code ec = GetTestEC();
        path ret = temp_directory_path(ec);
        assert(!ec);

        // Check that we return the same as above when no vars were defined.
        assert(ret == fallback);

        // Finally erase this env variable
        UnsetEnv(TC.name);
    }
}

int main(int, char**) {
    signature_test();
    basic_tests();
    return 0;
}
