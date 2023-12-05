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

// class recursive_directory_iterator

// recursive_directory_iterator begin(recursive_directory_iterator iter) noexcept;
// recursive_directory_iterator end(recursive_directory_iterator iter) noexcept;

#include <filesystem>
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void test_function_signatures()
{
    recursive_directory_iterator d;

    ASSERT_SAME_TYPE(decltype(begin(d)), recursive_directory_iterator);
    ASSERT_SAME_TYPE(decltype(begin(std::move(d))), recursive_directory_iterator);
    ASSERT_NOEXCEPT(begin(d));
    ASSERT_NOEXCEPT(begin(std::move(d)));

    ASSERT_SAME_TYPE(decltype(end(d)), recursive_directory_iterator);
    ASSERT_SAME_TYPE(decltype(end(std::move(d))), recursive_directory_iterator);
    ASSERT_NOEXCEPT(end(d));
    ASSERT_NOEXCEPT(end(std::move(d)));
}

static void test_ranged_for_loop()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    std::set<path> dir_contents(static_env.RecDirIterationList.begin(),
                                static_env.RecDirIterationList.end());

    std::error_code ec;
    recursive_directory_iterator it(testDir, ec);
    assert(!ec);

    for (auto& elem : it) {
        assert(dir_contents.erase(elem) == 1);
    }
    assert(dir_contents.empty());
}

int main(int, char**) {
    test_function_signatures();
    test_ranged_for_loop();

    return 0;
}
