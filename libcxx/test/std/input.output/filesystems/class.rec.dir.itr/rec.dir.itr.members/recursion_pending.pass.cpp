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

// bool recursion_pending() const;

#include <filesystem>
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void initial_value_test()
{
    static_test_env static_env;
    recursive_directory_iterator it(static_env.Dir);
    assert(it.recursion_pending() == true);
}

static void value_after_copy_construction_and_assignment_test()
{
    static_test_env static_env;
    recursive_directory_iterator rec_pending_it(static_env.Dir);
    recursive_directory_iterator no_rec_pending_it(static_env.Dir);
    no_rec_pending_it.disable_recursion_pending();

    { // copy construction
        recursive_directory_iterator it(rec_pending_it);
        assert(it.recursion_pending() == true);
        it.disable_recursion_pending();
        assert(rec_pending_it.recursion_pending() == true);

        recursive_directory_iterator it2(no_rec_pending_it);
        assert(it2.recursion_pending() == false);
    }
    { // copy assignment
        recursive_directory_iterator it(static_env.Dir);
        it.disable_recursion_pending();
        it = rec_pending_it;
        assert(it.recursion_pending() == true);
        it.disable_recursion_pending();
        assert(rec_pending_it.recursion_pending() == true);

        recursive_directory_iterator it2(static_env.Dir);
        it2 = no_rec_pending_it;
        assert(it2.recursion_pending() == false);
    }
    assert(rec_pending_it.recursion_pending() == true);
    assert(no_rec_pending_it.recursion_pending() == false);
}


static void value_after_move_construction_and_assignment_test()
{
    static_test_env static_env;
    recursive_directory_iterator rec_pending_it(static_env.Dir);
    recursive_directory_iterator no_rec_pending_it(static_env.Dir);
    no_rec_pending_it.disable_recursion_pending();

    { // move construction
        recursive_directory_iterator it_cp(rec_pending_it);
        recursive_directory_iterator it(std::move(it_cp));
        assert(it.recursion_pending() == true);

        recursive_directory_iterator it_cp2(no_rec_pending_it);
        recursive_directory_iterator it2(std::move(it_cp2));
        assert(it2.recursion_pending() == false);
    }
    { // copy assignment
        recursive_directory_iterator it(static_env.Dir);
        it.disable_recursion_pending();
        recursive_directory_iterator it_cp(rec_pending_it);
        it = std::move(it_cp);
        assert(it.recursion_pending() == true);

        recursive_directory_iterator it2(static_env.Dir);
        recursive_directory_iterator it_cp2(no_rec_pending_it);
        it2 = std::move(it_cp2);
        assert(it2.recursion_pending() == false);
    }
    assert(rec_pending_it.recursion_pending() == true);
    assert(no_rec_pending_it.recursion_pending() == false);
}

static void increment_resets_value()
{
    static_test_env static_env;
    const recursive_directory_iterator endIt;
    {
        recursive_directory_iterator it(static_env.Dir);
        it.disable_recursion_pending();
        assert(it.recursion_pending() == false);
        ++it;
        assert(it.recursion_pending() == true);
        assert(it.depth() == 0);
    }
    {
        recursive_directory_iterator it(static_env.Dir);
        it.disable_recursion_pending();
        assert(it.recursion_pending() == false);
        it++;
        assert(it.recursion_pending() == true);
        assert(it.depth() == 0);
    }
    {
        recursive_directory_iterator it(static_env.Dir);
        it.disable_recursion_pending();
        assert(it.recursion_pending() == false);
        std::error_code ec;
        it.increment(ec);
        assert(it.recursion_pending() == true);
        assert(it.depth() == 0);
    }
}

static void pop_does_not_reset_value()
{
    static_test_env static_env;
    const recursive_directory_iterator endIt;

    auto& DE0 = static_env.DirIterationList;
    std::set<path> notSeenDepth0(DE0.begin(), DE0.end());

    recursive_directory_iterator it(static_env.Dir);
    assert(it != endIt);

    while (it.depth() == 0) {
        notSeenDepth0.erase(it->path());
        ++it;
        assert(it != endIt);
    }
    assert(it.depth() == 1);
    it.disable_recursion_pending();
    it.pop();
    // Since the order of iteration is unspecified the pop() could result
    // in the end iterator. When this is the case it is undefined behavior
    // to call recursion_pending().
    if (it == endIt) {
        assert(notSeenDepth0.empty());
#if defined(_LIBCPP_VERSION)
        assert(it.recursion_pending() == false);
#endif
    } else {
        assert(! notSeenDepth0.empty());
        assert(it.recursion_pending() == false);
    }
}

int main(int, char**) {
    initial_value_test();
    value_after_copy_construction_and_assignment_test();
    value_after_move_construction_and_assignment_test();
    increment_resets_value();
    pop_does_not_reset_value();

    return 0;
}
