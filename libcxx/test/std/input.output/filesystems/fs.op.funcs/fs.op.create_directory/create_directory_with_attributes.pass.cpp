//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-create-symlinks
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: availability-filesystem-missing

// This test requires the dylib support introduced in e4ed349c7658.
// XFAIL: using-built-library-before-llvm-12

// <filesystem>

// bool create_directory(const path& p, const path& attr);
// bool create_directory(const path& p, const path& attr, error_code& ec) noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void test_signatures()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p, p)), bool);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p, p, ec)), bool);
    ASSERT_NOT_NOEXCEPT(fs::create_directory(p, p));
    ASSERT_NOEXCEPT(fs::create_directory(p, p, ec));
}

static void create_existing_directory()
{
    scoped_test_env env;
    const path dir = env.create_dir("dir1");
    const path dir2 = env.create_dir("dir2");

    const perms orig_p = status(dir).permissions();
    permissions(dir2, perms::none);

    std::error_code ec;
    assert(fs::create_directory(dir, dir2, ec) == false);
    assert(!ec);

    // Check that the permissions were unchanged
    assert(orig_p == status(dir).permissions());

    // Test throwing version
    assert(fs::create_directory(dir, dir2) == false);
}

// Windows doesn't have the concept of perms::none on directories.
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
static void create_directory_one_level()
{
    scoped_test_env env;
    // Remove setgid which mkdir would inherit
    permissions(env.test_root, perms::set_gid, perm_options::remove);

    const path dir = env.make_env_path("dir1");
    const path attr_dir = env.create_dir("dir2");
    permissions(attr_dir, perms::none);

    std::error_code ec;
    assert(fs::create_directory(dir, attr_dir, ec) == true);
    assert(!ec);
    assert(is_directory(dir));

    // Check that the new directory has the same permissions as attr_dir
    auto st = status(dir);
    assert(st.permissions() == perms::none);
}
#endif // TEST_WIN_NO_FILESYSTEM_PERMS_NONE

static void create_directory_multi_level()
{
    scoped_test_env env;
    const path dir = env.make_env_path("dir1/dir2");
    const path dir1 = env.make_env_path("dir1");
    const path attr_dir = env.create_dir("attr_dir");
    std::error_code ec = GetTestEC();
    assert(fs::create_directory(dir, attr_dir, ec) == false);
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));
    assert(!is_directory(dir));
    assert(!is_directory(dir1));
}

static void dest_is_file()
{
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    const path attr_dir = env.create_dir("attr_dir");
    std::error_code ec = GetTestEC();
    assert(fs::create_directory(file, attr_dir, ec) == false);
    assert(ec);
    assert(is_regular_file(file));
}

static void dest_part_is_file()
{
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    const path dir = env.make_env_path("file/dir1");
    const path attr_dir = env.create_dir("attr_dir");
    std::error_code ec = GetTestEC();
    assert(fs::create_directory(dir, attr_dir, ec) == false);
    assert(ec);
    assert(is_regular_file(file));
    assert(!exists(dir));
}

static void attr_dir_is_invalid() {
  scoped_test_env env;
  const path file = env.create_file("file", 42);
  const path dest = env.make_env_path("dir");
  const path dne = env.make_env_path("dne");
  {
    std::error_code ec = GetTestEC();
    assert(create_directory(dest, file, ec) == false);
    assert(ErrorIs(ec, std::errc::not_a_directory));
  }
  assert(!exists(dest));
  {
    std::error_code ec = GetTestEC();
    assert(create_directory(dest, dne, ec) == false);
    assert(ErrorIs(ec, std::errc::not_a_directory));
  }
}

static void dest_is_symlink_to_unexisting() {
  scoped_test_env env;
  const path attr_dir = env.create_dir("attr_dir");
  const path sym = env.create_symlink("dne_sym", "dne_sym_name");
  {
    std::error_code ec = GetTestEC();
    assert(create_directory(sym, attr_dir, ec) == false);
    assert(ec);
  }
}

static void dest_is_symlink_to_dir() {
  scoped_test_env env;
  const path dir = env.create_dir("dir");
  const path sym = env.create_directory_symlink(dir, "sym_name");
  const path attr_dir = env.create_dir("attr_dir");
  {
    std::error_code ec = GetTestEC();
    assert(create_directory(sym, attr_dir, ec) == false);
    assert(!ec);
  }
}

static void dest_is_symlink_to_file() {
  scoped_test_env env;
  const path file = env.create_file("file");
  const path sym = env.create_symlink(file, "sym_name");
  const path attr_dir = env.create_dir("attr_dir");
  {
    std::error_code ec = GetTestEC();
    assert(create_directory(sym, attr_dir, ec) == false);
    assert(ec);
  }
}

int main(int, char**) {
  test_signatures();
  create_existing_directory();
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
  create_directory_one_level();
#endif
  create_directory_multi_level();
  dest_is_file();
  dest_part_is_file();
  attr_dir_is_invalid();
  dest_is_symlink_to_unexisting();
  dest_is_symlink_to_dir();
  dest_is_symlink_to_file();

  return 0;
}
