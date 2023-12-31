//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_INPUT_OUTPUT_FILE_STREAMS_FSTREAMS_TEST_HELPERS_H
#define TEST_STD_INPUT_OUTPUT_FILE_STREAMS_FSTREAMS_TEST_HELPERS_H

#if _LIBCPP_STD_VER >= 26

#  include <cassert>
#  include <cstdio>
#  include <fstream>
#  include <filesystem>
#  include <utility>

#  if defined(_LIBCPP_WIN32API)
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#    include <io.h>
#    include <windows.h>
#  else
#    include <fcntl.h>
#  endif

#  include "platform_support.h"

#  if defined(_LIBCPP_WIN32API)
auto is_handle_valid([[maybe_unused]] HANDLE handle) {
  // TODO: Maybe test HANDLE with GetFileInformationByHandle??
  return true;
};
#  else
// POSIX
auto is_handle_valid(int fd) { return fcntl(fd, F_GETFL) != -1 || errno != EBADF; };
#  endif

template <typename CharT, typename StreamT>
void test_native_handle() {
  static_assert(
      std::is_same_v<typename std::basic_filebuf<CharT>::native_handle_type, typename StreamT::native_handle_type>);

  StreamT f;
  static_assert(noexcept(f.native_handle()));
  assert(!f.is_open());
  std::filesystem::path p = get_temp_file_name();
  f.open(p);
  assert(f.is_open());
  assert(f.native_handle() == f.rdbuf()->native_handle());
  assert(is_handle_valid(f.native_handle()));
  assert(is_handle_valid(std::as_const(f).native_handle()));
  static_assert(noexcept(f.native_handle()));
}

#endif // _LIBCPP_STD_VER >= 26

#endif // TEST_STD_INPUT_OUTPUT_FILE_STREAMS_FSTREAMS_TEST_HELPERS_H
