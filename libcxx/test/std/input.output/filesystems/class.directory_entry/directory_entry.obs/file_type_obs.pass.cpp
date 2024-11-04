//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Starting in Android N (API 24), SELinux policy prevents the shell user from
// creating a hard link.
// XFAIL: LIBCXX-ANDROID-FIXME && !android-device-api={{21|22|23}}

// <filesystem>

// class directory_entry

// file_status status() const;
// file_status status(error_code const&) const noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "filesystem_test_helper.h"

#include "test_macros.h"

static void file_dne() {
  using namespace fs;
  directory_entry p("dne");
}

static void signatures() {
  using namespace fs;
  const directory_entry e = {};
  std::error_code ec;
#define TEST_FUNC(name)                                                        \
  static_assert(std::is_same<decltype(e.name()), bool>::value,                 \
                "wrong return type");                                          \
  static_assert(noexcept(e.name()) == false, "should not be noexcept");        \
  static_assert(std::is_same<decltype(e.name(ec)), bool>::value,               \
                "wrong return type");                                          \
  static_assert(noexcept(e.name(ec)) == true, "should be noexcept")

  TEST_FUNC(exists);
  TEST_FUNC(is_block_file);
  TEST_FUNC(is_character_file);
  TEST_FUNC(is_directory);
  TEST_FUNC(is_fifo);
  TEST_FUNC(is_other);
  TEST_FUNC(is_regular_file);
  TEST_FUNC(is_socket);
  TEST_FUNC(is_symlink);

#undef TEST_FUNC
}

static void test_without_ec() {
  using namespace fs;
  using fs::directory_entry;
  using fs::file_status;
  using fs::path;

  scoped_test_env env;
  path f = env.create_file("foo", 42);
  path d = env.create_dir("dir");
  path hl = env.create_hardlink("foo", "hl");
  auto test_path = [=](const path &p) {
    directory_entry e(p);
    file_status st = status(p);
    file_status sym_st = symlink_status(p);
    fs::remove(p);
    assert(e.exists());
    assert(!exists(p));
    assert(e.exists() == exists(st));
    assert(e.is_block_file() == is_block_file(st));
    assert(e.is_character_file() == is_character_file(st));
    assert(e.is_directory() == is_directory(st));
    assert(e.is_fifo() == is_fifo(st));
    assert(e.is_other() == is_other(st));
    assert(e.is_regular_file() == is_regular_file(st));
    assert(e.is_socket() == is_socket(st));
    assert(e.is_symlink() == is_symlink(sym_st));
  };
  test_path(f);
  test_path(d);
  test_path(hl);
#ifndef _WIN32
  path fifo = env.create_fifo("fifo");
  test_path(fifo);
#endif
}

static void test_with_ec() {
  using namespace fs;
  using fs::directory_entry;
  using fs::file_status;
  using fs::path;

  scoped_test_env env;
  path f = env.create_file("foo", 42);
  path d = env.create_dir("dir");
  path hl = env.create_hardlink("foo", "hl");
  auto test_path = [=](const path &p) {
    directory_entry e(p);
    std::error_code status_ec = GetTestEC();
    std::error_code sym_status_ec = GetTestEC(1);
    file_status st = status(p, status_ec);
    file_status sym_st = symlink_status(p, sym_status_ec);
    fs::remove(p);
    std::error_code ec = GetTestEC(2);
    auto CheckEC = [&](std::error_code const& other_ec) {
      bool res = ec == other_ec;
      ec = GetTestEC(2);
      return res;
    };

    assert(e.exists(ec));
    assert(CheckEC(status_ec));
    assert(!exists(p));

    assert(e.exists(ec) == exists(st));
    assert(CheckEC(status_ec));

    assert(e.is_block_file(ec) == is_block_file(st));
    assert(CheckEC(status_ec));

    assert(e.is_character_file(ec) == is_character_file(st));
    assert(CheckEC(status_ec));

    assert(e.is_directory(ec) == is_directory(st));
    assert(CheckEC(status_ec));

    assert(e.is_fifo(ec) == is_fifo(st));
    assert(CheckEC(status_ec));

    assert(e.is_other(ec) == is_other(st));
    assert(CheckEC(status_ec));

    assert(e.is_regular_file(ec) == is_regular_file(st));
    assert(CheckEC(status_ec));

    assert(e.is_socket(ec) == is_socket(st));
    assert(CheckEC(status_ec));

    assert(e.is_symlink(ec) == is_symlink(sym_st));
    assert(CheckEC(sym_status_ec));
  };
  test_path(f);
  test_path(d);
  test_path(hl);
#ifndef _WIN32
  path fifo = env.create_fifo("fifo");
  test_path(fifo);
#endif
}

static void test_with_ec_dne() {
  using namespace fs;
  using fs::directory_entry;
  using fs::file_status;
  using fs::path;
  static_test_env static_env;
  for (auto p : {static_env.DNE, static_env.BadSymlink}) {

    directory_entry e(p);
    std::error_code status_ec = GetTestEC();
    std::error_code sym_status_ec = GetTestEC(1);
    file_status st = status(p, status_ec);
    file_status sym_st = symlink_status(p, sym_status_ec);
    std::error_code ec = GetTestEC(2);
    auto CheckEC = [&](std::error_code const& other_ec) {
      bool res = ec == other_ec;
      ec = GetTestEC(2);
      return res;
    };

    assert(e.exists(ec) == exists(st));
    assert(CheckEC(status_ec));

    assert(e.is_block_file(ec) == is_block_file(st));
    assert(CheckEC(status_ec));

    assert(e.is_character_file(ec) == is_character_file(st));
    assert(CheckEC(status_ec));

    assert(e.is_directory(ec) == is_directory(st));
    assert(CheckEC(status_ec));

    assert(e.is_fifo(ec) == is_fifo(st));
    assert(CheckEC(status_ec));

    assert(e.is_other(ec) == is_other(st));
    assert(CheckEC(status_ec));

    assert(e.is_regular_file(ec) == is_regular_file(st));
    assert(CheckEC(status_ec));

    assert(e.is_socket(ec) == is_socket(st));
    assert(CheckEC(status_ec));

    assert(e.is_symlink(ec) == is_symlink(sym_st));
    assert(CheckEC(sym_status_ec));
  }
}

#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
// Windows doesn't support setting perms::none to trigger failures
// reading directories.
static void test_with_ec_cannot_resolve() {
  using namespace fs;
  using fs::directory_entry;
  using fs::file_status;
  using fs::path;

  scoped_test_env env;
  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);
  const path file_out_of_dir = env.create_file("file2", 99);
  const path sym = env.create_symlink("file2", "dir/sym");

  perms old_perms = fs::status(dir).permissions();

  for (auto p : {file, sym}) {
    permissions(dir, old_perms);
    directory_entry e(p);

    permissions(dir, perms::none);
    std::error_code dummy_ec;
    e.refresh(dummy_ec);
    assert(dummy_ec);

    std::error_code status_ec = GetTestEC();
    std::error_code sym_status_ec = GetTestEC(1);
    file_status st = status(p, status_ec);
    file_status sym_st = symlink_status(p, sym_status_ec);
    std::error_code ec = GetTestEC(2);
    auto CheckEC = [&](std::error_code const& other_ec) {
      bool res = ec == other_ec;
      ec = GetTestEC(2);
      return res;
    };

    assert(e.exists(ec) == exists(st));
    assert(CheckEC(status_ec));

    assert(e.is_block_file(ec) == is_block_file(st));
    assert(CheckEC(status_ec));

    assert(e.is_character_file(ec) == is_character_file(st));
    assert(CheckEC(status_ec));

    assert(e.is_directory(ec) == is_directory(st));
    assert(CheckEC(status_ec));

    assert(e.is_fifo(ec) == is_fifo(st));
    assert(CheckEC(status_ec));

    assert(e.is_other(ec) == is_other(st));
    assert(CheckEC(status_ec));

    assert(e.is_regular_file(ec) == is_regular_file(st));
    assert(CheckEC(status_ec));

    assert(e.is_socket(ec) == is_socket(st));
    assert(CheckEC(status_ec));

    assert(e.is_symlink(ec) == is_symlink(sym_st));
    assert(CheckEC(sym_status_ec));
  }
}
#endif // TEST_WIN_NO_FILESYSTEM_PERMS_NONE

int main(int, char**) {
  file_dne();
  signatures();
  test_without_ec();
  test_with_ec();
  test_with_ec_dne();
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
  test_with_ec_cannot_resolve();
#endif

  return 0;
}
