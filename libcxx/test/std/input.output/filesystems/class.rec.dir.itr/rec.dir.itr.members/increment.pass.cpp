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

// On Android L, ~scoped_test_env() is unable to delete the temp dir using
// chmod+rm because chmod is too broken.
// XFAIL: LIBCXX-ANDROID-FIXME && android-device-api={{21|22}}

// <filesystem>

// class recursive_directory_iterator

// recursive_directory_iterator& operator++();
// recursive_directory_iterator& increment(error_code& ec) noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <set>
#include <cassert>

#include "assert_macros.h"
#include "test_macros.h"
#include "filesystem_test_helper.h"

using namespace fs;

static void test_increment_signatures()
{
    recursive_directory_iterator d; ((void)d);
    std::error_code ec; ((void)ec);

    ASSERT_SAME_TYPE(decltype(++d), recursive_directory_iterator&);
    ASSERT_NOT_NOEXCEPT(++d);

    ASSERT_SAME_TYPE(decltype(d.increment(ec)), recursive_directory_iterator&);
    ASSERT_NOT_NOEXCEPT(d.increment(ec));
}

static void test_prefix_increment()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const std::set<path> dir_contents(static_env.RecDirIterationList.begin(),
                                      static_env.RecDirIterationList.end());
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(testDir, ec);
    assert(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        assert(it != endIt);
        const path entry = *it;
        assert(unseen_entries.erase(entry) == 1);
        recursive_directory_iterator& it_ref = ++it;
        assert(&it_ref == &it);
    }

    assert(it == endIt);
}

static void test_postfix_increment()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    const std::set<path> dir_contents(static_env.RecDirIterationList.begin(),
                                      static_env.RecDirIterationList.end());
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(testDir, ec);
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
    const std::set<path> dir_contents(static_env.RecDirIterationList.begin(),
                                      static_env.RecDirIterationList.end());
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(testDir, ec);
    assert(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        assert(it != endIt);
        const path entry = *it;
        assert(unseen_entries.erase(entry) == 1);
        recursive_directory_iterator& it_ref = it.increment(ec);
        assert(!ec);
        assert(&it_ref == &it);
    }

    assert(it == endIt);
}

static void test_follow_symlinks()
{
    static_test_env static_env;
    const path testDir = static_env.Dir;
    auto const& IterList = static_env.RecDirFollowSymlinksIterationList;

    const std::set<path> dir_contents(IterList.begin(), IterList.end());
    const recursive_directory_iterator endIt{};

    std::error_code ec;
    recursive_directory_iterator it(testDir,
                              directory_options::follow_directory_symlink, ec);
    assert(!ec);

    std::set<path> unseen_entries = dir_contents;
    while (!unseen_entries.empty()) {
        assert(it != endIt);
        const path entry = *it;

        assert(unseen_entries.erase(entry) == 1);
        recursive_directory_iterator& it_ref = it.increment(ec);
        assert(!ec);
        assert(&it_ref == &it);
    }
    assert(it == endIt);
}

// Windows doesn't support setting perms::none to trigger failures
// reading directories.
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
static void access_denied_on_recursion_test_case()
{
    using namespace fs;
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
        assert(ec != GetTestEC());
        assert(!ec);
        while (it != endIt && it->path() != permDeniedDir)
            ++it;
        assert(it != endIt);
        assert(*it == permDeniedDir);

        it.increment(ec);
        assert(ec);
        assert(it == endIt);
    }
    // Same as above but test operator++().
    {
        std::error_code ec = GetTestEC();
        recursive_directory_iterator it(startDir, ec);
        assert(!ec);
        while (it != endIt && it->path() != permDeniedDir)
            ++it;
        assert(it != endIt);
        assert(*it == permDeniedDir);

        TEST_THROWS_TYPE(filesystem_error, ++it);
    }
    // Test that recursion resulting in a "EACCESS" error is ignored when the
    // correct options are given to the constructor.
    {
        std::error_code ec = GetTestEC();
        recursive_directory_iterator it(startDir, SkipEPerm, ec);
        assert(!ec);
        assert(it != endIt);

        bool seenOtherFile = false;
        if (*it == otherFile) {
            ++it;
            seenOtherFile = true;
            assert (it != endIt);
        }
        assert(*it == permDeniedDir);

        ec = GetTestEC();
        it.increment(ec);
        assert(!ec);

        if (seenOtherFile) {
            assert(it == endIt);
        } else {
            assert(it != endIt);
            assert(*it == otherFile);
        }
    }
    // Test that construction resulting in a "EACCESS" error is not ignored
    // by default.
    {
        std::error_code ec;
        recursive_directory_iterator it(permDeniedDir, ec);
        assert(ec);
        assert(it == endIt);
    }
    // Same as above but testing the throwing constructors
    {
        TEST_THROWS_TYPE(filesystem_error,
                           recursive_directory_iterator(permDeniedDir));
    }
    // Test that construction resulting in a "EACCESS" error constructs the
    // end iterator when the correct options are given.
    {
        std::error_code ec = GetTestEC();
        recursive_directory_iterator it(permDeniedDir, SkipEPerm, ec);
        assert(!ec);
        assert(it == endIt);
    }
}

// See llvm.org/PR35078
static void test_PR35078()
{
  using namespace fs;
    scoped_test_env env;
    const path testFiles[] = {
        env.create_dir("dir1"),
        env.create_dir("dir1/dir2"),
        env.create_dir("dir1/dir2/dir3"),
        env.create_file("dir1/file1"),
        env.create_file("dir1/dir2/dir3/file2")
    };
    const path startDir = testFiles[0];
    const path permDeniedDir = testFiles[1];
    const path nestedDir = testFiles[2];
    const path nestedFile = testFiles[3];

    // Change the permissions so we can no longer iterate
    permissions(permDeniedDir,
                perms::group_exec|perms::owner_exec|perms::others_exec,
                perm_options::remove);

    const std::errc eacess = std::errc::permission_denied;
    std::error_code ec = GetTestEC();

    const recursive_directory_iterator endIt;

    auto SetupState = [&](bool AllowEAccess, bool& SeenFile3) {
      SeenFile3 = false;
      auto Opts = AllowEAccess ? directory_options::skip_permission_denied
          : directory_options::none;
      recursive_directory_iterator it(startDir, Opts, ec);
      while (!ec && it != endIt && *it != nestedDir) {
        if (*it == nestedFile)
          SeenFile3 = true;
        it.increment(ec);
      }
      return it;
    };

    {
      bool SeenNestedFile = false;
      recursive_directory_iterator it = SetupState(false, SeenNestedFile);
      assert(it != endIt);
      assert(*it == nestedDir);
      ec = GetTestEC();
      it.increment(ec);
      assert(ec);
      assert(ErrorIs(ec, eacess));
      assert(it == endIt);
    }
    {
      bool SeenNestedFile = false;
      recursive_directory_iterator it = SetupState(true, SeenNestedFile);
      assert(it != endIt);
      assert(*it == nestedDir);
      ec = GetTestEC();
      it.increment(ec);
      assert(!ec);
      if (SeenNestedFile) {
        assert(it == endIt);
      } else {
        assert(it != endIt);
        assert(*it == nestedFile);
      }
    }
    {
      bool SeenNestedFile = false;
      recursive_directory_iterator it = SetupState(false, SeenNestedFile);
      assert(it != endIt);
      assert(*it == nestedDir);

      ExceptionChecker Checker(std::errc::permission_denied,
                               "recursive_directory_iterator::operator++()",
                               format_string("attempting recursion into \"%s\"",
                                             nestedDir.string().c_str()));
      TEST_VALIDATE_EXCEPTION(filesystem_error, Checker, ++it);
    }
}


// See llvm.org/PR35078
static void test_PR35078_with_symlink()
{
  using namespace fs;
    scoped_test_env env;
    const path testFiles[] = {
        env.create_dir("dir1"),
        env.create_file("dir1/file1"),
        env.create_dir("sym_dir"),
        env.create_dir("sym_dir/nested_sym_dir"),
        env.create_directory_symlink("sym_dir/nested_sym_dir", "dir1/dir2"),
        env.create_dir("sym_dir/dir1"),
        env.create_dir("sym_dir/dir1/dir2"),

    };
   // const unsigned TestFilesSize = sizeof(testFiles) / sizeof(testFiles[0]);
    const path startDir = testFiles[0];
    const path nestedFile = testFiles[1];
    const path permDeniedDir = testFiles[2];
    const path symDir = testFiles[4];

    // Change the permissions so we can no longer iterate
    permissions(permDeniedDir,
                perms::group_exec|perms::owner_exec|perms::others_exec,
                perm_options::remove);

    const std::errc eacess = std::errc::permission_denied;
    std::error_code ec = GetTestEC();

    const recursive_directory_iterator endIt;

    auto SetupState = [&](bool AllowEAccess, bool FollowSym, bool& SeenFile3) {
      SeenFile3 = false;
      auto Opts = AllowEAccess ? directory_options::skip_permission_denied
          : directory_options::none;
      if (FollowSym)
        Opts |= directory_options::follow_directory_symlink;
      recursive_directory_iterator it(startDir, Opts, ec);
      while (!ec && it != endIt && *it != symDir) {
        if (*it == nestedFile)
          SeenFile3 = true;
        it.increment(ec);
      }
      return it;
    };

    struct {
      bool SkipPermDenied;
      bool FollowSymlinks;
      bool ExpectSuccess;
    } TestCases[]  = {
        // Passing cases
        {false, false, true}, {true, true, true}, {true, false, true},
        // Failing cases
        {false, true, false}
    };
    for (auto TC : TestCases) {
      bool SeenNestedFile = false;
      recursive_directory_iterator it = SetupState(TC.SkipPermDenied,
                                                   TC.FollowSymlinks,
                                                   SeenNestedFile);
      assert(!ec);
      assert(it != endIt);
      assert(*it == symDir);
      ec = GetTestEC();
      it.increment(ec);
      if (TC.ExpectSuccess) {
        assert(!ec);
        if (SeenNestedFile) {
          assert(it == endIt);
        } else {
          assert(it != endIt);
          assert(*it == nestedFile);
        }
      } else {
        assert(ec);
        assert(ErrorIs(ec, eacess));
        assert(it == endIt);
      }
    }
}


// See llvm.org/PR35078
static void test_PR35078_with_symlink_file()
{
  using namespace fs;
    scoped_test_env env;
    const path testFiles[] = {
        env.create_dir("dir1"),
        env.create_dir("dir1/dir2"),
        env.create_file("dir1/file2"),
        env.create_dir("sym_dir"),
        env.create_dir("sym_dir/sdir1"),
        env.create_file("sym_dir/sdir1/sfile1"),
        env.create_symlink("sym_dir/sdir1/sfile1", "dir1/dir2/file1")
    };
    const unsigned TestFilesSize = sizeof(testFiles) / sizeof(testFiles[0]);
    const path startDir = testFiles[0];
    const path nestedDir = testFiles[1];
    const path nestedFile = testFiles[2];
    const path permDeniedDir = testFiles[3];
    const path symFile = testFiles[TestFilesSize - 1];

    // Change the permissions so we can no longer iterate
    permissions(permDeniedDir,
                perms::group_exec|perms::owner_exec|perms::others_exec,
                perm_options::remove);

    const std::errc eacess = std::errc::permission_denied;
    std::error_code ec = GetTestEC();

    const recursive_directory_iterator EndIt;

    auto SetupState = [&](bool AllowEAccess, bool FollowSym, bool& SeenNestedFile) {
      SeenNestedFile = false;
      auto Opts = AllowEAccess ? directory_options::skip_permission_denied
          : directory_options::none;
      if (FollowSym)
        Opts |= directory_options::follow_directory_symlink;
      recursive_directory_iterator it(startDir, Opts, ec);
      while (!ec && it != EndIt && *it != nestedDir) {
        if (*it == nestedFile)
          SeenNestedFile = true;
        it.increment(ec);
      }
      return it;
    };

    struct {
      bool SkipPermDenied;
      bool FollowSymlinks;
      bool ExpectSuccess;
    } TestCases[]  = {
        // Passing cases
        {false, false, true}, {true, true, true}, {true, false, true},
        // Failing cases
        {false, true, false}
    };
    for (auto TC : TestCases){
      bool SeenNestedFile = false;
      recursive_directory_iterator it = SetupState(TC.SkipPermDenied,
                                                   TC.FollowSymlinks,
                                                   SeenNestedFile);
      assert(!ec);
      assert(it != EndIt);
      assert(*it == nestedDir);
      ec = GetTestEC();
      it.increment(ec);
      assert(it != EndIt);
      assert(!ec);
      assert(*it == symFile);
      ec = GetTestEC();
      it.increment(ec);
      if (TC.ExpectSuccess) {
        if (!SeenNestedFile) {
          assert(!ec);
          assert(it != EndIt);
          assert(*it == nestedFile);
          ec = GetTestEC();
          it.increment(ec);
        }
        assert(!ec);
        assert(it == EndIt);
      } else {
        assert(ec);
        assert(ErrorIs(ec, eacess));
        assert(it == EndIt);
      }
    }
}
#endif // TEST_WIN_NO_FILESYSTEM_PERMS_NONE

int main(int, char**) {
    test_increment_signatures();
    test_prefix_increment();
    test_postfix_increment();
    test_increment_method();
    test_follow_symlinks();
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
    access_denied_on_recursion_test_case();
    test_PR35078();
    test_PR35078_with_symlink();
    test_PR35078_with_symlink_file();
#endif

    return 0;
}
