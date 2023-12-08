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

// This test requires the dylib support introduced in D92769.
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx{{10.15|11.0}}

// <filesystem>

// bool create_directories(const path& p);
// bool create_directories(const path& p, error_code& ec) noexcept;

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "assert_macros.h"
#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

static void test_signatures()
{
    const path p; ((void)p);
    std::error_code ec; ((void)ec);
    ASSERT_SAME_TYPE(decltype(fs::create_directories(p)), bool);
    ASSERT_SAME_TYPE(decltype(fs::create_directories(p, ec)), bool);
    ASSERT_NOT_NOEXCEPT(fs::create_directories(p));
    ASSERT_NOT_NOEXCEPT(fs::create_directories(p, ec));
}

static void create_existing_directory()
{
    scoped_test_env env;
    const path dir = env.create_dir("dir1");
    std::error_code ec;
    assert(fs::create_directories(dir, ec) == false);
    assert(!ec);
    assert(is_directory(dir));
}

static void create_directory_one_level()
{
    scoped_test_env env;
    const path dir = env.make_env_path("dir1");
    std::error_code ec;
    assert(fs::create_directories(dir, ec) == true);
    assert(!ec);
    assert(is_directory(dir));
}

static void create_directories_multi_level()
{
    scoped_test_env env;
    const path dir = env.make_env_path("dir1/dir2/dir3");
    std::error_code ec;
    assert(fs::create_directories(dir, ec) == true);
    assert(!ec);
    assert(is_directory(dir));
}

static void create_directory_symlinks() {
  scoped_test_env env;
  const path root = env.create_dir("dir");
  const path sym_dest_dead = env.make_env_path("dead");
  const path dead_sym = env.create_directory_symlink(sym_dest_dead, "dir/sym_dir");
  const path target = env.make_env_path("dir/sym_dir/foo");
  {
    std::error_code ec = GetTestEC();
    assert(create_directories(target, ec) == false);
    assert(ec);
    assert(ErrorIs(ec, std::errc::file_exists));
    assert(!exists(sym_dest_dead));
    assert(!exists(dead_sym));
  }
}

static void create_directory_through_symlinks() {
  scoped_test_env env;
  const path root = env.create_dir("dir");
  const path sym_dir = env.create_directory_symlink(root, "sym_dir");
  const path target = env.make_env_path("sym_dir/foo");
  const path resolved_target = env.make_env_path("dir/foo");
  assert(is_directory(sym_dir));
  {
    std::error_code ec = GetTestEC();
    assert(create_directories(target, ec) == true);
    assert(!ec);
    assert(is_directory(target));
    assert(is_directory(resolved_target));
  }
}

static void dest_is_file()
{
    scoped_test_env env;
    const path file = env.create_file("file", 42);
    std::error_code ec = GetTestEC();
    assert(fs::create_directories(file, ec) == false);
    assert(ec);
    assert(ErrorIs(ec, std::errc::file_exists));
    assert(is_regular_file(file));
}

static void dest_part_is_file()
{
    scoped_test_env env;
    const path file = env.create_file("file");
    const path dir = env.make_env_path("file/dir1");
    std::error_code ec = GetTestEC();
    assert(fs::create_directories(dir, ec) == false);
    assert(ec);
    assert(ErrorIs(ec, std::errc::not_a_directory));
    assert(is_regular_file(file));
    assert(!exists(dir));
}

static void dest_final_part_is_file()
{
    scoped_test_env env;
    env.create_dir("dir");
    const path file = env.create_file("dir/file");
    const path dir = env.make_env_path("dir/file/dir1");
    std::error_code ec = GetTestEC();
    assert(fs::create_directories(dir, ec) == false);
    assert(ec);
    assert(ErrorIs(ec, std::errc::not_a_directory));
    assert(is_regular_file(file));
    assert(!exists(dir));
}

static void dest_is_empty_path()
{
    std::error_code ec = GetTestEC();
    assert(fs::create_directories(fs::path{}, ec) == false);
    assert(ec);
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));
    ExceptionChecker Checker(path{}, std::errc::no_such_file_or_directory,
                             "create_directories");
    TEST_VALIDATE_EXCEPTION(filesystem_error, Checker,
                            fs::create_directories(path{}));
}

#ifdef _WIN32
static void nonexistent_root()
{
    std::error_code ec = GetTestEC();
    // If Q:\ doesn't exist, create_directories would try to recurse upwards
    // to parent_path() until it finds a directory that does exist. As the
    // whole path is the root name, parent_path() returns itself, and it
    // would recurse indefinitely, unless the recursion is broken.
    if (!exists("Q:\\"))
       assert(fs::create_directories("Q:\\", ec) == false);
    assert(fs::create_directories("\\\\nonexistentserver", ec) == false);
}
#endif // _WIN32

int main(int, char**) {
    test_signatures();
    create_existing_directory();
    create_directory_one_level();
    create_directories_multi_level();
    create_directory_symlinks();
    create_directory_through_symlinks();
    dest_is_file();
    dest_part_is_file();
    dest_final_part_is_file();
    dest_is_empty_path();
#ifdef _WIN32
    nonexistent_root();
#endif

    return 0;
}
