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

// directory_iterator& operator++();
// directory_iterator& increment(error_code& ec);

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"

using namespace fs;

static void test_increment_signatures()
{
    directory_iterator d; ((void)d);
    std::error_code ec; ((void)ec);

    ASSERT_SAME_TYPE(decltype(++d), directory_iterator&);
    ASSERT_NOT_NOEXCEPT(++d);

    ASSERT_SAME_TYPE(decltype(d.increment(ec)), directory_iterator&);
    ASSERT_NOT_NOEXCEPT(d.increment(ec));
}

static void test_prefix_increment()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const std::set<path> dir_contents(static_env.DirIterationList.begin(),
                                      static_env.DirIterationList.end());
    const directory_iterator endIt{};

    std::error_code ec;
    directory_iterator it(testDir, ec);
    assert(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        assert(it != endIt);
        const path entry = *it;
        assert(unseen_entries.erase(entry) == 1);
        directory_iterator& it_ref = ++it;
        assert(&it_ref == &it);
    }

    assert(it == endIt);
}

static void test_postfix_increment()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const std::set<path> dir_contents(static_env.DirIterationList.begin(),
                                      static_env.DirIterationList.end());
    const directory_iterator endIt{};

    std::error_code ec;
    directory_iterator it(testDir, ec);
    assert(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        assert(it != endIt);
        const path entry = *it;
        assert(unseen_entries.erase(entry) == 1);
        const path entry2 = *it++;
        assert(entry2 == entry);
    }

    assert(it == endIt);
}


static void test_increment_method()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const std::set<path> dir_contents(static_env.DirIterationList.begin(),
                                      static_env.DirIterationList.end());
    const directory_iterator endIt{};

    std::error_code ec;
    directory_iterator it(testDir, ec);
    assert(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        assert(it != endIt);
        const path entry = *it;
        assert(unseen_entries.erase(entry) == 1);
        directory_iterator& it_ref = it.increment(ec);
        assert(!ec);
        assert(&it_ref == &it);
    }

    assert(it == endIt);
}

int main(int, char**) {
    test_increment_signatures();
    test_prefix_increment();
    test_postfix_increment();
    test_increment_method();

    return 0;
}
