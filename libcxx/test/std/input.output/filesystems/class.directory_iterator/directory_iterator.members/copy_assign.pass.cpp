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

// class directory_iterator

// directory_iterator& operator=(directory_iterator const&);

#include <filesystem>
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void test_assignment_signature()
{
    using D = directory_iterator;
    static_assert(std::is_copy_assignable<D>::value, "");
}

static void test_copy_to_end_iterator()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;

    const directory_iterator from(testDir);
    assert(from != directory_iterator{});
    const path entry = *from;

    directory_iterator to{};
    to = from;
    assert(to == from);
    assert(*to == entry);
    assert(*from == entry);
}


static void test_copy_from_end_iterator()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;

    const directory_iterator from{};

    directory_iterator to(testDir);
    assert(to != directory_iterator{});

    to = from;
    assert(to == from);
    assert(to == directory_iterator{});
}

static void test_copy_valid_iterator()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const directory_iterator endIt{};

    directory_iterator it_obj(testDir);
    const directory_iterator& it = it_obj;
    assert(it != endIt);
    ++it_obj;
    assert(it != endIt);
    const path entry = *it;

    directory_iterator it2(testDir);
    assert(it2 != it);
    const path entry2 = *it2;
    assert(entry2 != entry);

    it2 = it;
    assert(it2 == it);
    assert(*it2 == entry);
}

static void test_returns_reference_to_self()
{
    const directory_iterator it;
    directory_iterator it2;
    directory_iterator& ref = (it2 = it);
    assert(&ref == &it2);
}

int main(int, char**) {
    test_assignment_signature();
    test_copy_to_end_iterator();
    test_copy_from_end_iterator();
    test_copy_valid_iterator();
    test_returns_reference_to_self();

    return 0;
}
