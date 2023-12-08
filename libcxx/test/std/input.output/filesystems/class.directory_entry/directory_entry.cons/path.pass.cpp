//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <filesystem>

// class directory_entry

// explicit directory_entry(const path);
// directory_entry(const path&, error_code& ec);

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "assert_macros.h"
#include "test_macros.h"
#include "filesystem_test_helper.h"
#include "test_convertible.h"
namespace fs = std::filesystem;

static void path_ctor() {
  using namespace fs;
  {
    static_assert(std::is_constructible<directory_entry, const path&>::value,
                  "directory_entry must be constructible from path");
    static_assert(
        !std::is_nothrow_constructible<directory_entry, const path&>::value,
        "directory_entry constructor should not be noexcept");
    static_assert(!std::is_convertible<path const&, directory_entry>::value,
                  "directory_entry constructor should be explicit");
  }
  {
    const path p("foo/bar/baz");
    const directory_entry e(p);
    assert(e.path() == p);
  }
}

static void path_ec_ctor() {
  static_test_env static_env;
  using namespace fs;
  {
    static_assert(
        std::is_constructible<directory_entry, const path&,
                              std::error_code&>::value,
        "directory_entry must be constructible from path and error_code");
    static_assert(!std::is_nothrow_constructible<directory_entry, const path&,
                                                 std::error_code&>::value,
                  "directory_entry constructor should not be noexcept");
    static_assert(
        test_convertible<directory_entry, const path&, std::error_code&>(),
        "directory_entry constructor should not be explicit");
  }
  {
    std::error_code ec = GetTestEC();
    const directory_entry e(static_env.File, ec);
    assert(e.path() == static_env.File);
    assert(!ec);
  }
  {
    const path p("foo/bar/baz");
    std::error_code ec = GetTestEC();
    const directory_entry e(p, ec);
    assert(e.path() == p);
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));
  }
}

static void path_ctor_calls_refresh() {
  using namespace fs;
  scoped_test_env env;
  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);
  const path sym = env.create_symlink("dir/file", "sym");

  {
    directory_entry ent(file);
    std::error_code ec = GetTestEC();
    directory_entry ent_ec(file, ec);
    assert(!ec);

    LIBCPP_ONLY(remove(file));

    assert(ent.exists());
    assert(ent_ec.exists());

    assert(ent.file_size() == 42);
    assert(ent_ec.file_size() == 42);
  }

  env.create_file("dir/file", 101);

  {
    directory_entry ent(sym);
    std::error_code ec = GetTestEC();
    directory_entry ent_ec(sym, ec);
    assert(!ec);

    LIBCPP_ONLY(remove(file));
    LIBCPP_ONLY(remove(sym));

    assert(ent.is_symlink());
    assert(ent_ec.is_symlink());

    assert(ent.is_regular_file());
    assert(ent_ec.is_regular_file());

    assert(ent.file_size() == 101);
    assert(ent_ec.file_size() == 101);
  }
}

static void path_ctor_dne() {
  using namespace fs;

  static_test_env static_env;

  {
    std::error_code ec = GetTestEC();
    directory_entry ent(static_env.DNE, ec);
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));
    assert(ent.path() == static_env.DNE);
  }
  // don't report dead symlinks as an error.
  {
    std::error_code ec = GetTestEC();
    directory_entry ent(static_env.BadSymlink, ec);
    assert(!ec);
    assert(ent.path() == static_env.BadSymlink);
  }
  // DNE does not cause the constructor to throw
  {
    directory_entry ent(static_env.DNE);
    assert(ent.path() == static_env.DNE);

    directory_entry ent_two(static_env.BadSymlink);
    assert(ent_two.path() == static_env.BadSymlink);
  }
}

static void path_ctor_cannot_resolve() {
  using namespace fs;
#ifdef _WIN32
  // Windows doesn't support setting perms::none to trigger failures
  // reading directories; test using a special inaccessible directory
  // instead.
  const path dir = GetWindowsInaccessibleDir();
  if (dir.empty())
    return;
  const path file = dir / "file";
  {
    std::error_code ec = GetTestEC();
    directory_entry ent(file, ec);
    assert(ErrorIs(ec, std::errc::no_such_file_or_directory));
    assert(ent.path() == file);
  }
  {
    TEST_DOES_NOT_THROW(directory_entry(file));
  }
#else
  scoped_test_env env;
  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);
  const path file_out_of_dir = env.create_file("file1", 101);
  const path sym_out_of_dir = env.create_symlink("dir/file", "sym");
  const path sym_in_dir = env.create_symlink("dir/file1", "dir/sym2");
  permissions(dir, perms::none);

  {
    std::error_code ec = GetTestEC();
    directory_entry ent(file, ec);
    assert(ErrorIs(ec, std::errc::permission_denied));
    assert(ent.path() == file);
  }
  {
    std::error_code ec = GetTestEC();
    directory_entry ent(sym_in_dir, ec);
    assert(ErrorIs(ec, std::errc::permission_denied));
    assert(ent.path() == sym_in_dir);
  }
  {
    std::error_code ec = GetTestEC();
    directory_entry ent(sym_out_of_dir, ec);
    assert(!ec);
    assert(ent.path() == sym_out_of_dir);
  }
  {
    TEST_DOES_NOT_THROW(directory_entry(file));
    TEST_DOES_NOT_THROW(directory_entry(sym_in_dir));
    TEST_DOES_NOT_THROW(directory_entry(sym_out_of_dir));
  }
#endif
}

int main(int, char**) {
  path_ctor();
  path_ec_ctor();
  path_ctor_calls_refresh();
  path_ctor_dne();
  path_ctor_cannot_resolve();

  return 0;
}
