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

// recursive_directory_iterator(recursive_directory_iterator&&) noexcept;

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
    using D = recursive_directory_iterator;
    static_assert(std::is_nothrow_move_constructible<D>::value, "");
}

static void test_move_end_iterator()
{
    const recursive_directory_iterator endIt;
    recursive_directory_iterator endIt2{};

    recursive_directory_iterator it(std::move(endIt2));
    assert(it == endIt);
    assert(endIt2 == endIt);
}

static void test_move_valid_iterator()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const recursive_directory_iterator endIt{};

    // build 'it' up with "interesting" non-default state so we can test
    // that it gets copied. We want to get 'it' into a state such that:
    //  it.options() != directory_options::none
    //  it.depth() != 0
    //  it.recursion_pending() != true
    const directory_options opts = directory_options::skip_permission_denied;
    recursive_directory_iterator it(testDir, opts);
    assert(it != endIt);
    while (it.depth() == 0) {
        ++it;
        assert(it != endIt);
    }
    it.disable_recursion_pending();
    assert(it.options() == opts);
    assert(it.depth() == 1);
    assert(it.recursion_pending() == false);
    const path entry = *it;

    // OPERATION UNDER TEST //
    const recursive_directory_iterator it2(std::move(it));
    // ------------------- //

    assert(it2 != endIt);
    assert(*it2 == entry);
    assert(it2.depth() == 1);
    assert(it2.recursion_pending() == false);
}

int main(int, char**) {
    test_constructor_signature();
    test_move_end_iterator();
    test_move_valid_iterator();

    return 0;
}
