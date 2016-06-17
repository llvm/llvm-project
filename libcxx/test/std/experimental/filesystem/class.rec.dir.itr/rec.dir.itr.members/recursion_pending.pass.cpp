//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

// class recursive_directory_iterator

// bool recursion_pending() const;

#include <experimental/filesystem>
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.hpp"
#include "filesystem_test_helper.hpp"

using namespace std::experimental::filesystem;

TEST_SUITE(recursive_directory_iterator_recursion_pending_tests)

TEST_CASE(initial_value_test)
{
    recursive_directory_iterator it(StaticEnv::Dir);
    TEST_REQUIRE(it.recursion_pending() == true);
}

TEST_CASE(value_after_copy_construction_and_assignment_test)
{
    recursive_directory_iterator rec_pending_it(StaticEnv::Dir);
    recursive_directory_iterator no_rec_pending_it(StaticEnv::Dir);
    no_rec_pending_it.disable_recursion_pending();

    { // copy construction
        recursive_directory_iterator it(rec_pending_it);
        TEST_CHECK(it.recursion_pending() == true);
        it.disable_recursion_pending();
        TEST_REQUIRE(rec_pending_it.recursion_pending() == true);

        recursive_directory_iterator it2(no_rec_pending_it);
        TEST_CHECK(it2.recursion_pending() == false);
    }
    { // copy assignment
        recursive_directory_iterator it(StaticEnv::Dir);
        it.disable_recursion_pending();
        it = rec_pending_it;
        TEST_CHECK(it.recursion_pending() == true);
        it.disable_recursion_pending();
        TEST_REQUIRE(rec_pending_it.recursion_pending() == true);

        recursive_directory_iterator it2(StaticEnv::Dir);
        it2 = no_rec_pending_it;
        TEST_CHECK(it2.recursion_pending() == false);
    }
    TEST_CHECK(rec_pending_it.recursion_pending() == true);
    TEST_CHECK(no_rec_pending_it.recursion_pending() == false);
}


TEST_CASE(value_after_move_construction_and_assignment_test)
{
    recursive_directory_iterator rec_pending_it(StaticEnv::Dir);
    recursive_directory_iterator no_rec_pending_it(StaticEnv::Dir);
    no_rec_pending_it.disable_recursion_pending();

    { // move construction
        recursive_directory_iterator it_cp(rec_pending_it);
        recursive_directory_iterator it(std::move(it_cp));
        TEST_CHECK(it.recursion_pending() == true);

        recursive_directory_iterator it_cp2(no_rec_pending_it);
        recursive_directory_iterator it2(std::move(it_cp2));
        TEST_CHECK(it2.recursion_pending() == false);
    }
    { // copy assignment
        recursive_directory_iterator it(StaticEnv::Dir);
        it.disable_recursion_pending();
        recursive_directory_iterator it_cp(rec_pending_it);
        it = std::move(it_cp);
        TEST_CHECK(it.recursion_pending() == true);

        recursive_directory_iterator it2(StaticEnv::Dir);
        recursive_directory_iterator it_cp2(no_rec_pending_it);
        it2 = std::move(it_cp2);
        TEST_CHECK(it2.recursion_pending() == false);
    }
    TEST_CHECK(rec_pending_it.recursion_pending() == true);
    TEST_CHECK(no_rec_pending_it.recursion_pending() == false);
}

TEST_CASE(increment_resets_value)
{
    const recursive_directory_iterator endIt;
    {
        recursive_directory_iterator it(StaticEnv::Dir);
        it.disable_recursion_pending();
        TEST_CHECK(it.recursion_pending() == false);
        ++it;
        TEST_CHECK(it.recursion_pending() == true);
        TEST_CHECK(it.depth() == 0);
    }
    {
        recursive_directory_iterator it(StaticEnv::Dir);
        it.disable_recursion_pending();
        TEST_CHECK(it.recursion_pending() == false);
        it++;
        TEST_CHECK(it.recursion_pending() == true);
        TEST_CHECK(it.depth() == 0);
    }
    {
        recursive_directory_iterator it(StaticEnv::Dir);
        it.disable_recursion_pending();
        TEST_CHECK(it.recursion_pending() == false);
        std::error_code ec;
        it.increment(ec);
        TEST_CHECK(it.recursion_pending() == true);
        TEST_CHECK(it.depth() == 0);
    }
}

TEST_CASE(pop_does_not_reset_value)
{
    const recursive_directory_iterator endIt;
    recursive_directory_iterator it(StaticEnv::Dir);

    while (it.depth() == 0) {
        ++it;
        TEST_REQUIRE(it != endIt);
    }
    it.disable_recursion_pending();
    it.pop();
    TEST_REQUIRE(it != endIt);
    TEST_CHECK(it.recursion_pending() == false);
}

TEST_SUITE_END()
