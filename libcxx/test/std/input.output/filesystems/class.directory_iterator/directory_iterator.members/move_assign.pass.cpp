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

// directory_iterator& operator=(directory_iterator const&);

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"

// The filesystem specification explicitly allows for self-move on
// the directory iterators. Turn off this warning so we can test it.
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wself-move")

using namespace fs;

static void test_assignment_signature()
{
    using D = directory_iterator;
    static_assert(std::is_nothrow_move_assignable<D>::value, "");
}

static void test_move_to_end_iterator()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;

    directory_iterator from(testDir);
    assert(from != directory_iterator{});
    const path entry = *from;

    directory_iterator to{};
    to = std::move(from);
    assert(to != directory_iterator{});
    assert(*to == entry);
}


static void test_move_from_end_iterator()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;

    directory_iterator from{};

    directory_iterator to(testDir);
    assert(to != from);

    to = std::move(from);
    assert(to == directory_iterator{});
    assert(from == directory_iterator{});
}

static void test_move_valid_iterator()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const directory_iterator endIt{};

    directory_iterator it(testDir);
    assert(it != endIt);
    ++it;
    assert(it != endIt);
    const path entry = *it;

    directory_iterator it2(testDir);
    assert(it2 != it);
    const path entry2 = *it2;
    assert(entry2 != entry);

    it2 = std::move(it);
    assert(it2 != directory_iterator{});
    assert(*it2 == entry);
}

static void test_returns_reference_to_self()
{
    directory_iterator it;
    directory_iterator it2;
    directory_iterator& ref = (it2 = it);
    assert(&ref == &it2);
}


static void test_self_move()
{
    static_test_env static_env;
    // Create two non-equal iterators that have exactly the same state.
    directory_iterator it(static_env.Dir);
    directory_iterator it2(static_env.Dir);
    ++it; ++it2;
    assert(it != it2);
    assert(*it2 == *it);

    it = std::move(it);
    assert(*it2 == *it);
}

int main(int, char**) {
    test_assignment_signature();
    test_move_to_end_iterator();
    test_move_from_end_iterator();
    test_move_valid_iterator();
    test_returns_reference_to_self();
    test_self_move();

    return 0;
}
