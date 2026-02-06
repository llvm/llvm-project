//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-filesystem
// ADDITIONAL_COMPILE_FLAGS: -I %{libcxx-dir}/src

// This test relies on calling functions from the libcxx internal headers
// of <filesystem>; the Windows implementation uses different
// internals and doesn't provide the same set_file_times function as for
// other platforms.
// UNSUPPORTED: windows

// This test assumes that time is stored as a 64 bit value when on MVS it is stored as 32 bit
// XFAIL: target={{.+}}-zos{{.*}}

// <filesystem>

// class directory_entry

#include <filesystem>
#include <type_traits>
#include <cassert>

#include "assert_macros.h"
#include "test_macros.h"
#include "filesystem_test_helper.h"

#include "filesystem/time_utils.h"

namespace fs = std::filesystem;
using namespace fs::detail;

static void last_write_time_not_representable_error() {
  using namespace fs;
  using namespace std::chrono;
  scoped_test_env env;
  const path dir = env.create_dir("dir");
  const path file = env.create_file("dir/file", 42);

  TimeSpec ToTime;
  ToTime.tv_sec = std::numeric_limits<decltype(ToTime.tv_sec)>::max();
  ToTime.tv_nsec = duration_cast<nanoseconds>(seconds(1)).count() - 1;

  std::array<TimeSpec, 2> TS = {ToTime, ToTime};

  file_time_type old_time = last_write_time(file);
  directory_entry ent(file);

  file_time_type start_time = file_time_type::clock::now() - hours(1);
  last_write_time(file, start_time);

  assert(ent.last_write_time() == old_time);

  bool IsRepresentable = true;
  file_time_type rep_value;
  {
    std::error_code ec;
    assert(!set_file_times(file, TS, ec));
    ec.clear();
    rep_value = last_write_time(file, ec);
    IsRepresentable = !bool(ec);
  }

  if (!IsRepresentable) {
    std::error_code rec = GetTestEC();
    ent.refresh(rec);
    assert(!rec);

    const std::errc expected_err = std::errc::value_too_large;

    std::error_code ec = GetTestEC();
    assert(ent.last_write_time(ec) == file_time_type::min());
    assert(ErrorIs(ec, expected_err));

    ec = GetTestEC();
    assert(last_write_time(file, ec) == file_time_type::min());
    assert(ErrorIs(ec, expected_err));

    ExceptionChecker CheckExcept(file, expected_err,
                                 "directory_entry::last_write_time");
    TEST_VALIDATE_EXCEPTION(filesystem_error, CheckExcept,
                            ent.last_write_time());

  } else {
    ent.refresh();

    std::error_code ec = GetTestEC();
    assert(ent.last_write_time(ec) == rep_value);
    assert(!ec);
  }
}

int main(int, char**) {
  last_write_time_not_representable_error();

  return 0;
}
