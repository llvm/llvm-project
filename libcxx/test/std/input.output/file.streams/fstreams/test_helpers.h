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
#  include <concepts>
#  include <cstdio>
#  include <fstream>
#  include <filesystem>
#  include <type_traits>
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
using HandleT = void*;

bool is_handle_valid([[HandleT handle) {
  if (LPBY_HANDLE_FILE_INFORMATION & pFileInformation; !GetFileInformationByHandle(handle, &lpFileInformation))
    return false;
  return true;
};
#  elif __has_include(<unistd.h>) // POSIX
using HandleT = int;

bool is_handle_valid(HandleT fd) { return fcntl(fd, F_GETFL) != -1 || errno != EBADF; };
#  else
#    error "Provide a native file handle!"
#  endif

template <typename CharT, typename StreamT>
void test_native_handle() {
  static_assert(
      std::is_same_v<typename std::basic_filebuf<CharT>::native_handle_type, typename StreamT::native_handle_type>);

  StreamT f;

  assert(!f.is_open());
  std::filesystem::path p = get_temp_file_name();
  f.open(p);
  assert(f.is_open());
  assert(f.native_handle() == f.rdbuf()->native_handle());
  std::same_as<HandleT> decltype(auto) handle = f.native_handle();
  assert(is_handle_valid(handle));
  std::same_as<HandleT> decltype(auto) const_handle = std::as_const(f).native_handle();
  assert(is_handle_valid(const_handle));
  static_assert(noexcept(f.native_handle()));
  static_assert(noexcept(std::as_const(f).native_handle()));
}

#endif // _LIBCPP_STD_VER >= 26

#endif // TEST_STD_INPUT_OUTPUT_FILE_STREAMS_FSTREAMS_TEST_HELPERS_H
