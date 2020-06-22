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

// directory_iterator(directory_iterator&&) noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(directory_iterator_move_construct_tests)

TEST_CASE(test_constructor_signature)
{
    using D = directory_iterator;
    static_assert(std::is_nothrow_move_constructible<D>::value, "");
}

TEST_CASE(test_move_end_iterator)
{
    const directory_iterator endIt;
    directory_iterator endIt2{};

    directory_iterator it(std::move(endIt2));
    TEST_CHECK(it == endIt);
    TEST_CHECK(endIt2 == endIt);
}

TEST_CASE(test_move_valid_iterator)
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const directory_iterator endIt{};

    directory_iterator it(testDir);
    TEST_REQUIRE(it != endIt);
    const path entry = *it;

    const directory_iterator it2(std::move(it));
    TEST_CHECK(*it2 == entry);

    TEST_CHECK(it == it2 || it == endIt);
}

TEST_SUITE_END()
