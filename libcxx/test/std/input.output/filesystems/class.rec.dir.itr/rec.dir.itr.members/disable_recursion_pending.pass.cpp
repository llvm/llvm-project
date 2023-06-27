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

// void disable_recursion_pending();

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"

using namespace fs;


// NOTE: The main semantics of disable_recursion_pending are tested
// in the 'recursion_pending()' tests.
static void basic_test()
{
    static_test_env static_env;
    recursive_directory_iterator it(static_env.Dir);
    assert(it.recursion_pending() == true);
    it.disable_recursion_pending();
    assert(it.recursion_pending() == false);
    it.disable_recursion_pending();
    assert(it.recursion_pending() == false);
}

int main(int, char**) {
    basic_test();

    return 0;
}
