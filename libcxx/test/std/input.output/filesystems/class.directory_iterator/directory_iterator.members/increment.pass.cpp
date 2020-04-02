//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FILE_DEPENDENCIES: ../../Inputs/static_test_env
// UNSUPPORTED: c++98, c++03

// <filesystem>

// class directory_iterator

// directory_iterator& operator++();
// directory_iterator& increment(error_code& ec);

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"
#include <iostream>

using namespace fs;

TEST_SUITE(directory_iterator_increment_tests)

TEST_CASE(test_increment_signatures)
{
    directory_iterator d; ((void)d);
    std::error_code ec; ((void)ec);

    ASSERT_SAME_TYPE(decltype(++d), directory_iterator&);
    ASSERT_NOT_NOEXCEPT(++d);

    ASSERT_SAME_TYPE(decltype(d.increment(ec)), directory_iterator&);
    ASSERT_NOT_NOEXCEPT(d.increment(ec));
}

TEST_CASE(test_prefix_increment)
{
    const path testDir = StaticEnv::Dir;
    const std::set<path> dir_contents(std::begin(StaticEnv::DirIterationList),
                                      std::end(  StaticEnv::DirIterationList));
    const directory_iterator endIt{};

    std::error_code ec;
    directory_iterator it(testDir, ec);
    TEST_REQUIRE(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        TEST_REQUIRE(it != endIt);
        const path entry = *it;
        TEST_REQUIRE(unseen_entries.erase(entry) == 1);
        directory_iterator& it_ref = ++it;
        TEST_CHECK(&it_ref == &it);
    }

    TEST_CHECK(it == endIt);
}

TEST_CASE(test_postfix_increment)
{
    const path testDir = StaticEnv::Dir;
    const std::set<path> dir_contents(std::begin(StaticEnv::DirIterationList),
                                      std::end(  StaticEnv::DirIterationList));
    const directory_iterator endIt{};

    std::error_code ec;
    directory_iterator it(testDir, ec);
    TEST_REQUIRE(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        TEST_REQUIRE(it != endIt);
        const path entry = *it;
        TEST_REQUIRE(unseen_entries.erase(entry) == 1);
        const path entry2 = *it++;
        TEST_CHECK(entry2 == entry);
    }

    TEST_CHECK(it == endIt);
}


TEST_CASE(test_increment_method)
{
    const path testDir = StaticEnv::Dir;
    const std::set<path> dir_contents(std::begin(StaticEnv::DirIterationList),
                                      std::end(  StaticEnv::DirIterationList));
    const directory_iterator endIt{};

    std::error_code ec;
    directory_iterator it(testDir, ec);
    TEST_REQUIRE(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        TEST_REQUIRE(it != endIt);
        const path entry = *it;
        TEST_REQUIRE(unseen_entries.erase(entry) == 1);
        directory_iterator& it_ref = it.increment(ec);
        TEST_REQUIRE(!ec);
        TEST_CHECK(&it_ref == &it);
    }

    TEST_CHECK(it == endIt);
}

TEST_SUITE_END()
