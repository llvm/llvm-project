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

// Starting in Android N (API 24), SELinux policy prevents the shell user from
// creating a hard link.
// XFAIL: LIBCXX-ANDROID-FIXME && !android-device-api={{21|22|23}}

// <filesystem>

// uintmax_t hard_link_count(const path& p);
// uintmax_t hard_link_count(const path& p, std::error_code& ec) noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void signature_test()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(hard_link_count(p)), std::uintmax_t);
    ASSERT_SAME_TYPE(decltype(hard_link_count(p, ec)), std::uintmax_t);
    ASSERT_NOT_NOEXCEPT(hard_link_count(p));
    ASSERT_NOEXCEPT(hard_link_count(p, ec));
}

static void hard_link_count_for_file()
{
    static_test_env static_env;
    assert(hard_link_count(static_env.File) == 1);
    std::error_code ec;
    assert(hard_link_count(static_env.File, ec) == 1);
}

static void hard_link_count_for_directory()
{
    static_test_env static_env;
    std::uintmax_t DirExpect = 3; // hard link from . .. and Dir2
    std::uintmax_t Dir3Expect = 2; // hard link from . ..
    std::uintmax_t DirExpectAlt = DirExpect;
    std::uintmax_t Dir3ExpectAlt = Dir3Expect;
#if defined(__APPLE__)
    // Filesystems formatted with case sensitive hfs+ behave unixish as
    // expected. Normal hfs+ filesystems report the number of directory
    // entries instead.
    DirExpectAlt = 5; // .  ..  Dir2  file1  file2
    Dir3Expect = 3; // .  ..  file5
#endif
    assert(hard_link_count(static_env.Dir) == DirExpect ||
               hard_link_count(static_env.Dir) == DirExpectAlt ||
               hard_link_count(static_env.Dir) == 1);
    assert(hard_link_count(static_env.Dir3) == Dir3Expect ||
               hard_link_count(static_env.Dir3) == Dir3ExpectAlt ||
               hard_link_count(static_env.Dir3) == 1);

    std::error_code ec;
    assert(hard_link_count(static_env.Dir, ec) == DirExpect ||
               hard_link_count(static_env.Dir, ec) == DirExpectAlt ||
               hard_link_count(static_env.Dir) == 1);
    assert(hard_link_count(static_env.Dir3, ec) == Dir3Expect ||
               hard_link_count(static_env.Dir3, ec) == Dir3ExpectAlt ||
               hard_link_count(static_env.Dir3) == 1);
}

static void hard_link_count_increments_test()
{
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    assert(hard_link_count(file) == 1);

    env.create_hardlink(file, "file_hl");
    assert(hard_link_count(file) == 2);
}


static void hard_link_count_error_cases()
{
    static_test_env static_env;
    const path testCases[] = {
        static_env.BadSymlink,
        static_env.DNE
    };
    const std::uintmax_t expect = static_cast<std::uintmax_t>(-1);
    for (auto& TC : testCases) {
        std::error_code ec;
        assert(hard_link_count(TC, ec) == expect);
        assert(ec);
    }
}

int main(int, char**) {
    signature_test();
    hard_link_count_for_file();
    hard_link_count_for_directory();
    hard_link_count_increments_test();
    hard_link_count_error_cases();

    return 0;
}
