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

// class recursive_directory_iterator

// void disable_recursion_pending();

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(recursive_directory_iterator_disable_recursion_pending_tests)

// NOTE: The main semantics of disable_recursion_pending are tested
// in the 'recursion_pending()' tests.
TEST_CASE(basic_test)
{
    recursive_directory_iterator it(StaticEnv::Dir);
    TEST_REQUIRE(it.recursion_pending() == true);
    it.disable_recursion_pending();
    TEST_CHECK(it.recursion_pending() == false);
    it.disable_recursion_pending();
    TEST_CHECK(it.recursion_pending() == false);
}

TEST_SUITE_END()
