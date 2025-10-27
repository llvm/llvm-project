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

// <filesystem>

// class directory_iterator

// directory_iterator(directory_iterator&&) noexcept;

#include <filesystem>
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void test_constructor_signature()
{
    using D = directory_iterator;
    static_assert(std::is_nothrow_move_constructible<D>::value, "");
}

static void test_move_end_iterator()
{
    const directory_iterator endIt;
    directory_iterator endIt2{};

    directory_iterator it(std::move(endIt2));
    assert(it == endIt);
    assert(endIt2 == endIt);
}

static void test_move_valid_iterator()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const directory_iterator endIt{};

    directory_iterator it(testDir);
    assert(it != endIt);
    const path entry = *it;

    const directory_iterator it2(std::move(it));
    assert(*it2 == entry);

    assert(it == it2 || it == endIt);
}

int main(int, char**) {
    test_constructor_signature();
    test_move_end_iterator();
    test_move_valid_iterator();

    return 0;
}
