//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// This test requires the dylib support introduced in D92769.
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.15

// <filesystem>

// bool create_directory(const path& p, const path& attr);
// bool create_directory(const path& p, const path& attr, error_code& ec) noexcept;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "rapid-cxx-test.h"
#include "filesystem_test_helper.h"

using namespace fs;

TEST_SUITE(filesystem_create_directory_test_suite)

TEST_CASE(test_signatures)
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p, p)), bool);
    ASSERT_SAME_TYPE(decltype(fs::create_directory(p, p, ec)), bool);
    ASSERT_NOT_NOEXCEPT(fs::create_directory(p, p));
    ASSERT_NOEXCEPT(fs::create_directory(p, p, ec));
}

TEST_CASE(create_existing_directory)
{
    scoped_test_env env;
    const path dir = env.create_dir("dir1");
    const path dir2 = env.create_dir("dir2");

    const perms orig_p = status(dir).permissions();
    permissions(dir2, perms::none);

    std::error_code ec;
    TEST_CHECK(fs::create_directory(dir, dir2, ec) == false);
    TEST_CHECK(!ec);

    // Check that the permissions were unchanged
    TEST_CHECK(orig_p == status(dir).permissions());

    // Test throwing version
    TEST_CHECK(fs::create_directory(dir, dir2) == false);
}

// Windows doesn't have the concept of perms::none on directories.
#ifndef TEST_WIN_NO_FILESYSTEM_PERMS_NONE
TEST_CASE(create_directory_one_level)
{
    scoped_test_env env;
    // Remove setgid which mkdir would inherit
    permissions(env.test_root, perms::set_gid, perm_options::remove);

    const path dir = env.make_env_path("dir1");
    const path attr_dir = env.create_dir("dir2");
    permissions(attr_dir, perms::none);

    std::error_code ec;
    TEST_CHECK(fs::create_directory(dir, attr_dir, ec) == true);
    TEST_CHECK(!ec);
    TEST_CHECK(is_directory(dir));

    // Check that the new directory has the same permissions as attr_dir
    auto st = status(dir);
    TEST_CHECK(st.permissions() == perms::none);
}
#endif

TEST_CASE(create_directory_multi_level)
{
    scoped_test_env env;
    const path dir = env.make_env_path("dir1/dir2");
    const path dir1 = env.make_env_path("dir1");
    const path attr_dir = env.create_dir("attr_dir");
    std::error_code ec = GetTestEC();
    TEST_CHECK(fs::create_directory(dir, attr_dir, ec) == false);
    TEST_CHECK(ErrorIs(ec, std::errc::no_such_file_or_directory));
    TEST_CHECK(!is_directory(dir));
    TEST_CHECK(!is_directory(dir1));
}

TEST_CASE(dest_is_file)
{
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    const path attr_dir = env.create_dir("attr_dir");
    std::error_code ec = GetTestEC();
    TEST_CHECK(fs::create_directory(file, attr_dir, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(is_regular_file(file));
}

TEST_CASE(dest_part_is_file)
{
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    const path dir = env.make_env_path("file/dir1");
    const path attr_dir = env.create_dir("attr_dir");
    std::error_code ec = GetTestEC();
    TEST_CHECK(fs::create_directory(dir, attr_dir, ec) == false);
    TEST_CHECK(ec);
    TEST_CHECK(is_regular_file(file));
    TEST_CHECK(!exists(dir));
}

TEST_CASE(attr_dir_is_invalid) {
  scoped_test_env env;
  const path file = env.create_file("file", 42);
  const path dest = env.make_env_path("dir");
  const path dne = env.make_env_path("dne");
  {
    std::error_code ec = GetTestEC();
    TEST_CHECK(create_directory(dest, file, ec) == false);
    TEST_CHECK(ErrorIs(ec, std::errc::not_a_directory));
  }
  TEST_REQUIRE(!exists(dest));
  {
    std::error_code ec = GetTestEC();
    TEST_CHECK(create_directory(dest, dne, ec) == false);
    TEST_CHECK(ErrorIs(ec, std::errc::not_a_directory));
  }
}

TEST_CASE(dest_is_symlink_to_unexisting) {
  scoped_test_env env;
  const path attr_dir = env.create_dir("attr_dir");
  const path sym = env.create_symlink("dne_sym", "dne_sym_name");
  {
    std::error_code ec = GetTestEC();
    TEST_CHECK(create_directory(sym, attr_dir, ec) == false);
    TEST_CHECK(ec);
  }
}

TEST_CASE(dest_is_symlink_to_dir) {
  scoped_test_env env;
  const path dir = env.create_dir("dir");
  const path sym = env.create_directory_symlink(dir, "sym_name");
  const path attr_dir = env.create_dir("attr_dir");
  {
    std::error_code ec = GetTestEC();
    TEST_CHECK(create_directory(sym, attr_dir, ec) == false);
    TEST_CHECK(!ec);
  }
}

TEST_CASE(dest_is_symlink_to_file) {
  scoped_test_env env;
  const path file = env.create_file("file");
  const path sym = env.create_symlink(file, "sym_name");
  const path attr_dir = env.create_dir("attr_dir");
  {
    std::error_code ec = GetTestEC();
    TEST_CHECK(create_directory(sym, attr_dir, ec) == false);
    TEST_CHECK(ec);
  }
}

TEST_SUITE_END()
