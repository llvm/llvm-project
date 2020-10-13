//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class directory_iterator

// directory_iterator begin(directory_iterator iter) noexcept;
// directory_iterator end(directory_iterator iter) noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(directory_iterator_begin_end_tests)

TEST_CASE(test_function_signatures)
{
    directory_iterator d; ((void)d);

    ASSERT_SAME_TYPE(decltype(begin(d)), directory_iterator);
    ASSERT_NOEXCEPT(begin(std::move(d)));

    ASSERT_SAME_TYPE(decltype(end(d)), directory_iterator);
    ASSERT_NOEXCEPT(end(std::move(d)));
}

TEST_CASE(test_ranged_for_loop)
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    std::set<path> dir_contents(static_env.DirIterationList.begin(),
                                static_env.DirIterationList.end());

    std::error_code ec;
    directory_iterator it(testDir, ec);
    TEST_REQUIRE(!ec);

    for (auto& elem : it) {
        TEST_CHECK(dir_contents.erase(elem) == 1);
    }
    TEST_CHECK(dir_contents.empty());
}

TEST_SUITE_END()
