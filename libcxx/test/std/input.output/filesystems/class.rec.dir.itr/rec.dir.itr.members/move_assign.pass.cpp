//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class recursive_directory_iterator

// recursive_directory_iterator& operator=(recursive_directory_iterator const&);

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

recursive_directory_iterator createInterestingIterator(const static_test_env &static_env)
    // Create an "interesting" iterator where all fields are
    // in a non-default state. The returned 'it' is in a
    // state such that:
    //   it.options() == directory_options::skip_permission_denied
    //   it.depth() == 1
    //   it.recursion_pending() == true
{
    const path testDir = static_env.Dir;
    const recursive_directory_iterator endIt;
    recursive_directory_iterator it(testDir,
                                    directory_options::skip_permission_denied);
    assert(it != endIt);
    while (it.depth() != 1) {
        ++it;
        assert(it != endIt);
    }
    assert(it.depth() == 1);
    it.disable_recursion_pending();
    return it;
}

recursive_directory_iterator createDifferentInterestingIterator(const static_test_env &static_env)
    // Create an "interesting" iterator where all fields are
    // in a non-default state. The returned 'it' is in a
    // state such that:
    //   it.options() == directory_options::follow_directory_symlink
    //   it.depth() == 2
    //   it.recursion_pending() == false
{
    const path testDir = static_env.Dir;
    const recursive_directory_iterator endIt;
    recursive_directory_iterator it(testDir,
                                    directory_options::follow_directory_symlink);
    assert(it != endIt);
    while (it.depth() != 2) {
        ++it;
        assert(it != endIt);
    }
    assert(it.depth() == 2);
    return it;
}


static void test_assignment_signature()
{
    using D = recursive_directory_iterator;
    static_assert(std::is_nothrow_move_assignable<D>::value, "");
}


static void test_move_to_end_iterator()
{
    static_test_env static_env;
    const recursive_directory_iterator endIt;

    recursive_directory_iterator from = createInterestingIterator(static_env);
    const recursive_directory_iterator from_copy(from);
    const path entry = *from;

    recursive_directory_iterator to;
    to = std::move(from);
    assert(to != endIt);
    assert(*to == entry);
    assert(to.options() == from_copy.options());
    assert(to.depth() == from_copy.depth());
    assert(to.recursion_pending() == from_copy.recursion_pending());
    assert(from == endIt || from == to);
}


static void test_move_from_end_iterator()
{
    static_test_env static_env;
    recursive_directory_iterator from;
    recursive_directory_iterator to = createInterestingIterator(static_env);

    to = std::move(from);
    assert(to == from);
    assert(to == recursive_directory_iterator{});
}

static void test_move_valid_iterator()
{
    static_test_env static_env;
    const recursive_directory_iterator endIt;

    recursive_directory_iterator it = createInterestingIterator(static_env);
    const recursive_directory_iterator it_copy(it);
    const path entry = *it;

    recursive_directory_iterator it2 = createDifferentInterestingIterator(static_env);
    const recursive_directory_iterator it2_copy(it2);
    assert(it2 != it);
    assert(it2.options() != it.options());
    assert(it2.depth() != it.depth());
    assert(it2.recursion_pending() != it.recursion_pending());
    assert(*it2 != entry);

    it2 = std::move(it);
    assert(it2 != it2_copy && it2 != endIt);
    assert(it2.options() == it_copy.options());
    assert(it2.depth() == it_copy.depth());
    assert(it2.recursion_pending() == it_copy.recursion_pending());
    assert(*it2 == entry);
    assert(it == endIt || it == it2);
}

static void test_returns_reference_to_self()
{
    recursive_directory_iterator it;
    recursive_directory_iterator it2;
    recursive_directory_iterator& ref = (it2 = std::move(it));
    assert(&ref == &it2);
}

static void test_self_move()
{
    static_test_env static_env;
    // Create two non-equal iterators that have exactly the same state.
    recursive_directory_iterator it = createInterestingIterator(static_env);
    recursive_directory_iterator it2 = createInterestingIterator(static_env);
    assert(it != it2);
    assert(it2.options()           == it.options());
    assert(it2.depth()             == it.depth());
    assert(it2.recursion_pending() == it.recursion_pending());
    assert(*it2 == *it);

    it = std::move(it);
    assert(it2.options()           == it.options());
    assert(it2.depth()             == it.depth());
    assert(it2.recursion_pending() == it.recursion_pending());
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
