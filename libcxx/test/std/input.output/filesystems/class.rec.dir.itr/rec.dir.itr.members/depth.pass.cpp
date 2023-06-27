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

// int depth() const

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "assert_macros.h"
#include "test_macros.h"
#include "filesystem_test_helper.h"

using namespace fs;

static void test_depth()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const path DirDepth1 = static_env.Dir2;
    const path DirDepth2 = static_env.Dir3;
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(testDir, ec);
    assert(!ec);
    assert(it.depth() == 0);

    bool seen_d1, seen_d2;
    seen_d1 = seen_d2 = false;

    while (it != endIt) {
        const path entry = *it;
        const path parent = entry.parent_path();
        if (parent == testDir) {
            assert(it.depth() == 0);
        } else if (parent == DirDepth1) {
            assert(it.depth() == 1);
            seen_d1 = true;
        } else if (parent == DirDepth2) {
            assert(it.depth() == 2);
            seen_d2 = true;
        } else {
            assert(!"Unexpected depth while iterating over static env");
        }
        ++it;
    }
    assert(seen_d1 && seen_d2);
    assert(it == endIt);
}

int main(int, char**) {
    test_depth();

    return 0;
}
