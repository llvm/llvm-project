//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-create-symlinks
// UNSUPPORTED: c++03, c++11, c++14

// The string reported on errors changed, which makes those tests fail when run
// against already-released libc++'s.
// XFAIL: stdlib=system && target={{.+}}-apple-macosx{{10.15|11.0}}

// <filesystem>

// class directory_entry

// file_time_type last_write_time() const;
// file_time_type last_write_time(error_code const&) const noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "assert_macros.h"
#include "filesystem_test_helper.h"
#include "test_macros.h"
namespace fs = std::filesystem;

static void signatures() {
  using namespace fs;
  {
    const fs::directory_entry e = {};
    std::error_code ec;
    static_assert(std::is_same<decltype(e.last_write_time()), file_time_type>::value,
                  "");
    static_assert(std::is_same<decltype(e.last_write_time(ec)), file_time_type>::value,
                  "");
    static_assert(noexcept(e.last_write_time()) == false, "");
    static_assert(noexcept(e.last_write_time(ec)) == true, "");
  }
}

static void basic() {
  using namespace fs;

  scoped_test_env env;
  const path file = env.create_file("file", 42);
  const path dir = env.create_dir("dir");
  const path sym = env.create_symlink("file", "sym");

  {
    directory_entry ent(file);
    file_time_type expect = last_write_time(ent);

    // Remove the file to show that the results were already in the cache.
    LIBCPP_ONLY(remove(file));

    std::error_code ec = GetTestEC();
    assert(ent.last_write_time(ec) == expect);
    assert(!ec);
  }
  {
    directory_entry ent(dir);
    file_time_type expect = last_write_time(ent);

    LIBCPP_ONLY(remove(dir));

    std::error_code ec = GetTestEC();
    assert(ent.last_write_time(ec) == expect);
    assert(!ec);
  }
  env.create_file("file", 99);
  {
    directory_entry ent(sym);
    file_time_type expect = last_write_time(sym);

    std::error_code ec = GetTestEC();
    assert(ent.last_write_time(ec) == expect);
    assert(!ec);
  }
}

static void error_reporting() {
  using namespace fs;

  static_test_env static_env;
  scoped_test_env env;

  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);
  const path file_out_of_dir = env.create_file("file2", 101);
  const path sym_out_of_dir = env.create_symlink("dir/file", "sym");
  const path sym_in_dir = env.create_symlink("file2", "dir/sym2");

#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
  const perms old_perms = status(dir).permissions();
#endif

  // test a file which doesn't exist
  {
    directory_entry ent;

    std::error_code ec = GetTestEC();
    ent.assign(static_env.DNE, ec);
    assert(ent.path() == static_env.DNE);
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));

    ec = GetTestEC();
    assert(ent.last_write_time(ec) == file_time_type::min());
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));

    ExceptionChecker Checker(static_env.DNE,
                             std::errc::no_such_file_or_directory,
                             "directory_entry::last_write_time");
    TEST_VALIDATE_EXCEPTION(filesystem_error, Checker, ent.last_write_time());
  }
  // test a dead symlink
  {
    directory_entry ent;

    std::error_code ec = GetTestEC();
    file_time_type expect_bad = last_write_time(static_env.BadSymlink, ec);
    assert(expect_bad == file_time_type::min());
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));

    ec = GetTestEC();
    ent.assign(static_env.BadSymlink, ec);
    assert(ent.path() == static_env.BadSymlink);
    assert(!ec);

    ec = GetTestEC();
    assert(ent.last_write_time(ec) == expect_bad);
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));

    ExceptionChecker Checker(static_env.BadSymlink,
                             std::errc::no_such_file_or_directory,
                             "directory_entry::last_write_time");
    TEST_VALIDATE_EXCEPTION(filesystem_error, Checker, ent.last_write_time());
  }
  // Windows doesn't support setting perms::none to trigger failures
  // reading directories.
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
  // test a file w/o appropriate permissions.
  {
    directory_entry ent;
    file_time_type expect_good = last_write_time(file);
    permissions(dir, perms::none);

    std::error_code ec = GetTestEC();
    ent.assign(file, ec);
    assert(ent.path() == file);
    assert(ErrorIs(ec, std::errc::permission_denied));

    ec = GetTestEC();
    assert(ent.last_write_time(ec) == file_time_type::min());
    assert(ErrorIs(ec, std::errc::permission_denied));

    ExceptionChecker Checker(file, std::errc::permission_denied,
                             "last_write_time");
    TEST_VALIDATE_EXCEPTION(filesystem_error, Checker, ent.last_write_time());

    permissions(dir, old_perms);
    ec = GetTestEC();
    assert(ent.last_write_time(ec) == expect_good);
    assert(!ec);
    TEST_DOES_NOT_THROW(ent.last_write_time());
  }
  permissions(dir, old_perms);
  // test a symlink w/o appropriate permissions.
  {
    directory_entry ent;
    file_time_type expect_good = last_write_time(sym_in_dir);
    permissions(dir, perms::none);

    std::error_code ec = GetTestEC();
    ent.assign(sym_in_dir, ec);
    assert(ent.path() == sym_in_dir);
    assert(ErrorIs(ec, std::errc::permission_denied));

    ec = GetTestEC();
    assert(ent.last_write_time(ec) == file_time_type::min());
    assert(ErrorIs(ec, std::errc::permission_denied));

    ExceptionChecker Checker(sym_in_dir, std::errc::permission_denied,
                             "last_write_time");
    TEST_VALIDATE_EXCEPTION(filesystem_error, Checker, ent.last_write_time());

    permissions(dir, old_perms);
    ec = GetTestEC();
    assert(ent.last_write_time(ec) == expect_good);
    assert(!ec);
    TEST_DOES_NOT_THROW(ent.last_write_time());
  }
  permissions(dir, old_perms);
  // test a symlink to a file w/o appropriate permissions
  {
    directory_entry ent;
    file_time_type expect_good = last_write_time(sym_out_of_dir);
    permissions(dir, perms::none);

    std::error_code ec = GetTestEC();
    ent.assign(sym_out_of_dir, ec);
    assert(ent.path() == sym_out_of_dir);
    assert(!ec);

    ec = GetTestEC();
    assert(ent.last_write_time(ec) == file_time_type::min());
    assert(ErrorIs(ec, std::errc::permission_denied));

    ExceptionChecker Checker(sym_out_of_dir, std::errc::permission_denied,
                             "last_write_time");
    TEST_VALIDATE_EXCEPTION(filesystem_error, Checker, ent.last_write_time());

    permissions(dir, old_perms);
    ec = GetTestEC();
    assert(ent.last_write_time(ec) == expect_good);
    assert(!ec);
    TEST_DOES_NOT_THROW(ent.last_write_time());
  }
#endif
}

int main(int, char**) {
  signatures();
  basic();
  error_reporting();

  return 0;
}
