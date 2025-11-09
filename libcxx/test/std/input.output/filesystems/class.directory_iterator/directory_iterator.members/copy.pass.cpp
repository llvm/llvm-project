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

// directory_iterator(directory_iterator const&);

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
    static_assert(std::is_copy_constructible<D>::value, "");
}

static void test_copy_end_iterator()
{
    const directory_iterator endIt;
    directory_iterator it(endIt);
    assert(it == endIt);
}

static void test_copy_valid_iterator()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const directory_iterator endIt{};

    const directory_iterator it(testDir);
    assert(it != endIt);
    const path entry = *it;

    const directory_iterator it2(it);
    assert(it2 == it);
    assert(*it2 == entry);
    assert(*it == entry);
}

int main(int, char**) {
    test_constructor_signature();
    test_copy_end_iterator();
    test_copy_valid_iterator();

    return 0;
}
