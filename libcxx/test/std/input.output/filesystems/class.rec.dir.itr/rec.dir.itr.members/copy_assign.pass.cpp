//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// class recursive_directory_iterator

// recursive_directory_iterator& operator=(recursive_directory_iterator const&);

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"

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

static void test_assignment_signature() {
    using D = recursive_directory_iterator;
    static_assert(std::is_copy_assignable<D>::value, "");
}

static void test_copy_to_end_iterator()
{
    static_test_env static_env;
    const recursive_directory_iterator endIt;

    const recursive_directory_iterator from = createInterestingIterator(static_env);
    const path entry = *from;

    recursive_directory_iterator to;
    to = from;
    assert(to == from);
    assert(*to == entry);
    assert(to.options() == from.options());
    assert(to.depth() == from.depth());
    assert(to.recursion_pending() == from.recursion_pending());
}


static void test_copy_from_end_iterator()
{
    static_test_env static_env;
    const recursive_directory_iterator from;
    recursive_directory_iterator to = createInterestingIterator(static_env);

    to = from;
    assert(to == from);
    assert(to == recursive_directory_iterator{});
}

static void test_copy_valid_iterator()
{
    static_test_env static_env;
    const recursive_directory_iterator endIt;

    const recursive_directory_iterator it = createInterestingIterator(static_env);
    const path entry = *it;

    recursive_directory_iterator it2 = createDifferentInterestingIterator(static_env);
    assert(it2                   != it);
    assert(it2.options()           != it.options());
    assert(it2.depth()             != it.depth());
    assert(it2.recursion_pending() != it.recursion_pending());
    assert(*it2                    != entry);

    it2 = it;
    assert(it2                   == it);
    assert(it2.options()           == it.options());
    assert(it2.depth()             == it.depth());
    assert(it2.recursion_pending() == it.recursion_pending());
    assert(*it2                    == entry);
}

static void test_returns_reference_to_self()
{
    const recursive_directory_iterator it;
    recursive_directory_iterator it2;
    recursive_directory_iterator& ref = (it2 = it);
    assert(&ref == &it2);
}

static void test_self_copy()
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

    // perform a self-copy and check that the state still matches the
    // other unmodified iterator.
    recursive_directory_iterator const& cit = it;
    it = cit;
    assert(it2.options()           == it.options());
    assert(it2.depth()             == it.depth());
    assert(it2.recursion_pending() == it.recursion_pending());
    assert(*it2 == *it);
}

int main(int, char**) {
    test_assignment_signature();
    test_copy_to_end_iterator();
    test_copy_from_end_iterator();
    test_copy_valid_iterator();
    test_returns_reference_to_self();
    test_self_copy();

    return 0;
}
