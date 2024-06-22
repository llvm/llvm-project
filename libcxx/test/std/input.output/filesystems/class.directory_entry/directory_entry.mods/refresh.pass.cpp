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

// directory_entry& operator=(directory_entry const&) = default;
// directory_entry& operator=(directory_entry&&) noexcept = default;
// void assign(path const&);
// void replace_filename(path const&);

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "assert_macros.h"
#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;

static void test_refresh_method() {
  using namespace fs;
  {
    directory_entry e;
    static_assert(noexcept(e.refresh()) == false,
                  "operation cannot be noexcept");
    static_assert(std::is_same<decltype(e.refresh()), void>::value,
                  "operation must return void");
  }
  {
    directory_entry e;
    e.refresh();
    assert(!e.exists());
  }
}

static void test_refresh_ec_method() {
  using namespace fs;
  {
    directory_entry e;
    std::error_code ec;
    static_assert(noexcept(e.refresh(ec)), "operation should be noexcept");
    static_assert(std::is_same<decltype(e.refresh(ec)), void>::value,
                  "operation must return void");
  }
  {
    directory_entry e;
    std::error_code ec = GetTestEC();
    e.refresh(ec);
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));
  }
}

#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
// Windows doesn't support setting perms::none to trigger failures
// reading directories.
static void refresh_on_file_dne() {
  using namespace fs;
  scoped_test_env env;
  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);

  const perms old_perms = status(dir).permissions();

  // test file doesn't exist
  {
    directory_entry ent(file);
    remove(file);
    assert(ent.exists());

    ent.refresh();

    permissions(dir, perms::none);
    assert(!ent.exists());
  }
  permissions(dir, old_perms);
  env.create_file("dir/file", 101);
  {
    directory_entry ent(file);
    remove(file);
    assert(ent.exists());

    std::error_code ec = GetTestEC();
    ent.refresh(ec);
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));

    permissions(dir, perms::none);
    assert(!ent.exists());
  }
}
#endif // TEST_WIN_NO_FILESYSTEM_PERMS_NONE

void remove_if_exists(const fs::path& p) {
  std::error_code ec;
  remove(p, ec);
}

static void refresh_on_bad_symlink() {
  using namespace fs;
  scoped_test_env env;
  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);
  const path sym = env.create_symlink("dir/file", "sym");

  const perms old_perms = status(dir).permissions();

  // test file doesn't exist
  {
    directory_entry ent(sym);
    LIBCPP_ONLY(remove(file));
    assert(ent.is_symlink());
    assert(ent.is_regular_file());
    assert(ent.exists());

    remove_if_exists(file);
    ent.refresh();

    LIBCPP_ONLY(permissions(dir, perms::none));
    assert(ent.is_symlink());
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
    assert(!ent.is_regular_file());
    assert(!ent.exists());
#endif
  }
  permissions(dir, old_perms);
  env.create_file("dir/file", 101);
  {
    directory_entry ent(sym);
    LIBCPP_ONLY(remove(file));
    assert(ent.is_symlink());
    assert(ent.is_regular_file());
    assert(ent.exists());

    remove_if_exists(file);

    std::error_code ec = GetTestEC();
    ent.refresh(ec);
    assert(!ec); // we don't report bad symlinks as an error.

    LIBCPP_ONLY(permissions(dir, perms::none));
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
    assert(!ent.exists());
#endif
  }
}

#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
// Windows doesn't support setting perms::none to trigger failures
// reading directories.
static void refresh_cannot_resolve() {
  using namespace fs;
  scoped_test_env env;
  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);
  const path file_out_of_dir = env.create_file("file1", 99);
  const path sym_out_of_dir = env.create_symlink("dir/file", "sym");
  const path sym_in_dir = env.create_symlink("file1", "dir/sym1");
  perms old_perms = status(dir).permissions();

  {
    directory_entry ent(file);
    permissions(dir, perms::none);

    assert(ent.is_regular_file());

    std::error_code ec = GetTestEC();
    ent.refresh(ec);

    assert(ErrorIs(ec, std::errc::permission_denied));
    assert(ent.path() == file);

    ExceptionChecker Checker(file, std::errc::permission_denied,
                             "directory_entry::refresh");
    TEST_VALIDATE_EXCEPTION(filesystem_error, Checker, ent.refresh());
  }
  permissions(dir, old_perms);
  {
    directory_entry ent(sym_in_dir);
    permissions(dir, perms::none);
    assert(ent.is_symlink());

    std::error_code ec = GetTestEC();
    ent.refresh(ec);
    assert(ErrorIs(ec, std::errc::permission_denied));
    assert(ent.path() == sym_in_dir);

    ExceptionChecker Checker(sym_in_dir, std::errc::permission_denied,
                             "directory_entry::refresh");
    TEST_VALIDATE_EXCEPTION(filesystem_error, Checker, ent.refresh());
  }
  permissions(dir, old_perms);
  {
    directory_entry ent(sym_out_of_dir);
    permissions(dir, perms::none);
    assert(ent.is_symlink());

    // Failure to resolve the linked entity due to permissions is not
    // reported as an error.
    std::error_code ec = GetTestEC();
    ent.refresh(ec);
    assert(!ec);
    assert(ent.is_symlink());

    ec = GetTestEC();
    assert(ent.exists(ec) == false);
    assert(ErrorIs(ec, std::errc::permission_denied));
    assert(ent.path() == sym_out_of_dir);
  }
  permissions(dir, old_perms);
  {
    directory_entry ent_file(file);
    directory_entry ent_sym(sym_in_dir);
    directory_entry ent_sym2(sym_out_of_dir);
    permissions(dir, perms::none);
    ((void)ent_file);
    ((void)ent_sym);

    TEST_THROWS_TYPE(filesystem_error, ent_file.refresh());
    TEST_THROWS_TYPE(filesystem_error, ent_sym.refresh());
  }
}
#endif // TEST_WIN_NO_FILESYSTEM_PERMS_NONE

static void refresh_doesnt_throw_on_dne_but_reports_it() {
  using namespace fs;
  scoped_test_env env;

  const path file = env.create_file("file1", 42);
  const path sym = env.create_symlink("file1", "sym");

  {
    directory_entry ent(file);
    assert(ent.file_size() == 42);

    remove(file);

    std::error_code ec = GetTestEC();
    ent.refresh(ec);
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));
    TEST_DOES_NOT_THROW(ent.refresh());

    ec = GetTestEC();
    assert(ent.file_size(ec) == std::uintmax_t(-1));
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));

    // doesn't throw!
    //
    //
    //
    //TEST_THROWS_TYPE(filesystem_error, ent.file_size());
  }
  env.create_file("file1", 99);
  {
    directory_entry ent(sym);
    assert(ent.is_symlink());
    assert(ent.is_regular_file());
    assert(ent.file_size() == 99);

    remove(file);

    std::error_code ec = GetTestEC();
    ent.refresh(ec);
    assert(!ec);

    ec = GetTestEC();
    assert(ent.file_size(ec) == std::uintmax_t(-1));
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));

    TEST_THROWS_TYPE(filesystem_error, ent.file_size());
  }
}

#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
// Windows doesn't support setting perms::none to trigger failures
// reading directories.
static void access_cache_after_refresh_fails() {
  using namespace fs;
  scoped_test_env env;
  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);
  const path file_out_of_dir = env.create_file("file1", 101);
  const path sym = env.create_symlink("dir/file", "sym");
  const path sym_in_dir = env.create_symlink("dir/file", "dir/sym2");

  const perms old_perms = status(dir).permissions();

#define CHECK_ACCESS(func, expect)                                             \
  ec = GetTestEC();                                                            \
  assert(ent.func(ec) == expect);                                          \
  assert(ErrorIs(ec, std::errc::permission_denied))

  // test file doesn't exist
  {
    directory_entry ent(file);

    assert(!ent.is_symlink());
    assert(ent.is_regular_file());
    assert(ent.exists());

    permissions(dir, perms::none);
    std::error_code ec = GetTestEC();
    ent.refresh(ec);
    assert(ErrorIs(ec, std::errc::permission_denied));

    CHECK_ACCESS(exists, false);
    CHECK_ACCESS(is_symlink, false);
    CHECK_ACCESS(last_write_time, file_time_type::min());
    CHECK_ACCESS(hard_link_count, std::uintmax_t(-1));
  }
  permissions(dir, old_perms);
  {
    directory_entry ent(sym_in_dir);
    assert(ent.is_symlink());
    assert(ent.is_regular_file());
    assert(ent.exists());

    permissions(dir, perms::none);
    std::error_code ec = GetTestEC();
    ent.refresh(ec);
    assert(ErrorIs(ec, std::errc::permission_denied));

    CHECK_ACCESS(exists, false);
    CHECK_ACCESS(is_symlink, false);
    CHECK_ACCESS(last_write_time, file_time_type::min());
    CHECK_ACCESS(hard_link_count, std::uintmax_t(-1));
  }
  permissions(dir, old_perms);
  {
    directory_entry ent(sym);
    assert(ent.is_symlink());
    assert(ent.is_regular_file());
    assert(ent.exists());

    permissions(dir, perms::none);
    std::error_code ec = GetTestEC();
    ent.refresh(ec);
    assert(!ec);
    assert(ent.is_symlink());

    CHECK_ACCESS(exists, false);
    CHECK_ACCESS(is_regular_file, false);
    CHECK_ACCESS(last_write_time, file_time_type::min());
    CHECK_ACCESS(hard_link_count, std::uintmax_t(-1));
  }
#undef CHECK_ACCESS
}
#endif // TEST_WIN_NO_FILESYSTEM_PERMS_NONE

int main(int, char**) {
  test_refresh_method();
  test_refresh_ec_method();
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
  refresh_on_file_dne();
#endif
  refresh_on_bad_symlink();
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
  refresh_cannot_resolve();
#endif
  refresh_doesnt_throw_on_dne_but_reports_it();
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
  access_cache_after_refresh_fails();
#endif
  return 0;
}
