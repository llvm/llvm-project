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

// recursive_directory_iterator& operator++();
// recursive_directory_iterator& increment(error_code& ec) noexcept;

#include <experimental/filesystem>
#include <type_traits>
#include <set>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.hpp"
#include "filesystem_test_helper.hpp"
#include <iostream>

using namespace std::experimental::filesystem;

TEST_SUITE(recursive_directory_iterator_increment_tests)

TEST_CASE(test_increment_signatures)
{
    using D = recursive_directory_iterator;
    recursive_directory_iterator d; ((void)d);
    std::error_code ec; ((void)ec);

    ASSERT_SAME_TYPE(decltype(++d), recursive_directory_iterator&);
    ASSERT_NOT_NOEXCEPT(++d);

    ASSERT_SAME_TYPE(decltype(d.increment(ec)), recursive_directory_iterator&);
    ASSERT_NOEXCEPT(d.increment(ec));
}

TEST_CASE(test_prefix_increment)
{
    const path testDir = StaticEnv::Dir;
    const std::set<path> dir_contents(std::begin(StaticEnv::RecDirIterationList),
                                      std::end(  StaticEnv::RecDirIterationList));
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(testDir, ec);
    TEST_REQUIRE(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        TEST_REQUIRE(it != endIt);
        const path entry = *it;
        TEST_REQUIRE(unseen_entries.erase(entry) == 1);
        recursive_directory_iterator& it_ref = ++it;
        TEST_CHECK(&it_ref == &it);
    }

    TEST_CHECK(it == endIt);
}

TEST_CASE(test_postfix_increment)
{
    const path testDir = StaticEnv::Dir;
    const std::set<path> dir_contents(std::begin(StaticEnv::RecDirIterationList),
                                      std::end(  StaticEnv::RecDirIterationList));
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(testDir, ec);
    TEST_REQUIRE(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        TEST_REQUIRE(it != endIt);
        const path entry = *it;
        TEST_REQUIRE(unseen_entries.erase(entry) == 1);
        const path entry2 = *it++;
        TEST_CHECK(entry2 == entry);
    }
    TEST_CHECK(it == endIt);
}


TEST_CASE(test_increment_method)
{
    const path testDir = StaticEnv::Dir;
    const std::set<path> dir_contents(std::begin(StaticEnv::RecDirIterationList),
                                      std::end(  StaticEnv::RecDirIterationList));
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(testDir, ec);
    TEST_REQUIRE(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        TEST_REQUIRE(it != endIt);
        const path entry = *it;
        TEST_REQUIRE(unseen_entries.erase(entry) == 1);
        recursive_directory_iterator& it_ref = it.increment(ec);
        TEST_REQUIRE(!ec);
        TEST_CHECK(&it_ref == &it);
    }

    TEST_CHECK(it == endIt);
}

TEST_CASE(test_follow_symlinks)
{
    const path testDir = StaticEnv::Dir;
    auto const& IterList = StaticEnv::RecDirFollowSymlinksIterationList;

    const std::set<path> dir_contents(std::begin(IterList), std::end(IterList));
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(testDir,
                              directory_options::follow_directory_symlink, ec);
    TEST_REQUIRE(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        TEST_REQUIRE(it != endIt);
        const path entry = *it;

        TEST_REQUIRE(unseen_entries.erase(entry) == 1);
        recursive_directory_iterator& it_ref = it.increment(ec);
        TEST_REQUIRE(!ec);
        TEST_CHECK(&it_ref == &it);
    }
    TEST_CHECK(it == endIt);
}

TEST_CASE(access_denied_on_recursion_test_case)
{
    using namespace std::experimental::filesystem;
    scoped_test_env env;
    const path testFiles[] = {
        env.create_dir("dir1"),
        env.create_dir("dir1/dir2"),
        env.create_file("dir1/dir2/file1"),
        env.create_file("dir1/file2")
    };
    const path startDir = testFiles[0];
    const path permDeniedDir = testFiles[1];
    const path otherFile = testFiles[3];
    auto SkipEPerm = directory_options::skip_permission_denied;

    // Change the permissions so we can no longer iterate
    permissions(permDeniedDir, perms::none);

    const recursive_directory_iterator endIt;

    // Test that recursion resulting in a "EACCESS" error is not ignored
    // by default.
    {
        std::error_code ec = GetTestEC();
        recursive_directory_iterator it(startDir, ec);
        TEST_REQUIRE(ec != GetTestEC());
        TEST_REQUIRE(!ec);
        while (it != endIt && it->path() != permDeniedDir)
            ++it;
        TEST_REQUIRE(it != endIt);
        TEST_REQUIRE(*it == permDeniedDir);

        it.increment(ec);
        TEST_CHECK(ec);
        TEST_CHECK(it == endIt);
    }
    // Same as above but test operator++().
    {
        std::error_code ec = GetTestEC();
        recursive_directory_iterator it(startDir, ec);
        TEST_REQUIRE(!ec);
        while (it != endIt && it->path() != permDeniedDir)
            ++it;
        TEST_REQUIRE(it != endIt);
        TEST_REQUIRE(*it == permDeniedDir);

        TEST_REQUIRE_THROW(filesystem_error, ++it);
    }
    // Test that recursion resulting in a "EACCESS" error is ignored when the
    // correct options are given to the constructor.
    {
        std::error_code ec = GetTestEC();
        recursive_directory_iterator it(startDir, SkipEPerm, ec);
        TEST_REQUIRE(!ec);
        TEST_REQUIRE(it != endIt);

        bool seenOtherFile = false;
        if (*it == otherFile) {
            ++it;
            seenOtherFile = true;
            TEST_REQUIRE (it != endIt);
        }
        TEST_REQUIRE(*it == permDeniedDir);

        ec = GetTestEC();
        it.increment(ec);
        TEST_REQUIRE(!ec);

        if (seenOtherFile) {
            TEST_CHECK(it == endIt);
        } else {
            TEST_CHECK(it != endIt);
            TEST_CHECK(*it == otherFile);
        }
    }
    // Test that construction resulting in a "EACCESS" error is not ignored
    // by default.
    {
        std::error_code ec;
        recursive_directory_iterator it(permDeniedDir, ec);
        TEST_REQUIRE(ec);
        TEST_REQUIRE(it == endIt);
    }
    // Same as above but testing the throwing constructors
    {
        TEST_REQUIRE_THROW(filesystem_error,
                           recursive_directory_iterator(permDeniedDir));
    }
    // Test that construction resulting in a "EACCESS" error constructs the
    // end iterator when the correct options are given.
    {
        std::error_code ec = GetTestEC();
        recursive_directory_iterator it(permDeniedDir, SkipEPerm, ec);
        TEST_REQUIRE(!ec);
        TEST_REQUIRE(it == endIt);
    }
}

TEST_SUITE_END()
