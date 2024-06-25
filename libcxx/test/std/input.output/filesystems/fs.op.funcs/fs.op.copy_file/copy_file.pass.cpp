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

// The string reported on errors changed, which makes those tests fail when run
// against already-released libc++'s.
// XFAIL: stdlib=system && target={{.+}}-apple-macosx{{10.15|11.0}}

// Starting in Android N (API 24), SELinux policy prevents the shell user from
// creating a FIFO file.
// XFAIL: LIBCXX-ANDROID-FIXME && !android-device-api={{21|22|23}}

// <filesystem>

// bool copy_file(const path& from, const path& to);
// bool copy_file(const path& from, const path& to, error_code& ec) noexcept;
// bool copy_file(const path& from, const path& to, copy_options options);
// bool copy_file(const path& from, const path& to, copy_options options, error_code& ec) noexcept;

#include <filesystem>
#include <type_traits>
#include <chrono>
#include <cassert>

#include "assert_macros.h"
#include "test_macros.h"
#include "filesystem_test_helper.h"
namespace fs = std::filesystem;
using namespace fs;

using CO = fs::copy_options;

static void test_signatures() {
  const path p;
  ((void)p);
  const copy_options opts{};
  ((void)opts);
  std::error_code ec;
  ((void)ec);
  ASSERT_SAME_TYPE(decltype(fs::copy_file(p, p)), bool);
  ASSERT_SAME_TYPE(decltype(fs::copy_file(p, p, opts)), bool);
  ASSERT_SAME_TYPE(decltype(fs::copy_file(p, p, ec)), bool);
  ASSERT_SAME_TYPE(decltype(fs::copy_file(p, p, opts, ec)), bool);
  ASSERT_NOT_NOEXCEPT(fs::copy_file(p, p));
  ASSERT_NOT_NOEXCEPT(fs::copy_file(p, p, opts));
  ASSERT_NOT_NOEXCEPT(fs::copy_file(p, p, ec));
  ASSERT_NOT_NOEXCEPT(fs::copy_file(p, p, opts, ec));
}

static void test_error_reporting() {

  scoped_test_env env;
  const path file = env.create_file("file1", 42);
  const path file2 = env.create_file("file2", 55);

  { // exists(to) && equivalent(to, from)
    std::error_code ec;
    assert(fs::copy_file(file, file, copy_options::overwrite_existing,
                             ec) == false);
    assert(ErrorIs(ec, std::errc::file_exists));
    ExceptionChecker Checker(file, file, std::errc::file_exists, "copy_file");
    TEST_VALIDATE_EXCEPTION(filesystem_error, Checker, copy_file(file, file, copy_options::overwrite_existing));

  }
  { // exists(to) && !(skip_existing | overwrite_existing | update_existing)
    std::error_code ec;
    assert(fs::copy_file(file, file2, ec) == false);
    assert(ErrorIs(ec, std::errc::file_exists));
    ExceptionChecker Checker(file, file, std::errc::file_exists, "copy_file");
    TEST_VALIDATE_EXCEPTION(filesystem_error, Checker, copy_file(file, file, copy_options::overwrite_existing));

  }
}

#ifndef _WIN32
static void non_regular_file_test() {
  scoped_test_env env;
  const path fifo = env.create_fifo("fifo");
  const path dest = env.make_env_path("dest");
  const path file = env.create_file("file", 42);

  {
    std::error_code ec = GetTestEC();
    assert(fs::copy_file(fifo, dest, ec) == false);
    assert(ErrorIs(ec, std::errc::not_supported));
    assert(!exists(dest));
  }
  {
    std::error_code ec = GetTestEC();
    assert(fs::copy_file(file, fifo, copy_options::overwrite_existing,
                               ec) == false);
    assert(ErrorIs(ec, std::errc::not_supported));
    assert(is_fifo(fifo));
  }

}
#endif // _WIN32

static void test_attributes_get_copied() {
  scoped_test_env env;
  const path file = env.create_file("file1", 42);
  const path dest = env.make_env_path("file2");
  (void)status(file);
  perms new_perms = perms::owner_read;
  permissions(file, new_perms);
  std::error_code ec = GetTestEC();
  assert(fs::copy_file(file, dest, ec) == true);
  assert(!ec);
  auto new_st = status(dest);
  assert(new_st.permissions() == NormalizeExpectedPerms(new_perms));
}

static void copy_dir_test() {
  scoped_test_env env;
  const path file = env.create_file("file1", 42);
  const path dest = env.create_dir("dir1");
  std::error_code ec = GetTestEC();
  assert(fs::copy_file(file, dest, ec) == false);
  assert(ec);
  assert(ec != GetTestEC());
  ec = GetTestEC();
  assert(fs::copy_file(dest, file, ec) == false);
  assert(ec);
  assert(ec != GetTestEC());
}

static void copy_file() {
  scoped_test_env env;
  const path file = env.create_file("file1", 42);

  { // !exists(to)
    const path dest = env.make_env_path("dest1");
    std::error_code ec = GetTestEC();

    assert(fs::copy_file(file, dest, ec) == true);
    assert(!ec);
    assert(file_size(dest) == 42);
  }
  { // exists(to) && overwrite_existing
    const path dest = env.create_file("dest2", 55);
    permissions(dest, perms::all);
    permissions(file,
                perms::group_write | perms::owner_write | perms::others_write,
                perm_options::remove);

    std::error_code ec = GetTestEC();
    assert(fs::copy_file(file, dest, copy_options::overwrite_existing,
                               ec) == true);
    assert(!ec);
    assert(file_size(dest) == 42);
    assert(status(dest).permissions() == status(file).permissions());
  }
  { // exists(to) && update_existing
    using Sec = std::chrono::seconds;
    const path older = env.create_file("older_file", 1);

    SleepFor(Sec(2));
    const path from = env.create_file("update_from", 55);

    SleepFor(Sec(2));
    const path newer = env.create_file("newer_file", 2);

    std::error_code ec = GetTestEC();
    assert(
        fs::copy_file(from, older, copy_options::update_existing, ec) == true);
    assert(!ec);
    assert(file_size(older) == 55);

    assert(
        fs::copy_file(from, newer, copy_options::update_existing, ec) == false);
    assert(!ec);
    assert(file_size(newer) == 2);
  }
  { // skip_existing
    const path file2 = env.create_file("file2", 55);
    std::error_code ec = GetTestEC();
    assert(fs::copy_file(file, file2, copy_options::skip_existing, ec) ==
                 false);
    assert(!ec);
    assert(file_size(file2) == 55);
  }
}

int main(int, char**) {
  test_signatures();
  test_error_reporting();
#ifndef _WIN32
  non_regular_file_test();
#endif
  test_attributes_get_copied();
  copy_dir_test();
  copy_file();

  return 0;
}
