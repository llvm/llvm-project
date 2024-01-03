//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_INPUT_OUTPUT_FILE_STREAMS_FSTREAMS_TEST_HELPERS_H
#define TEST_STD_INPUT_OUTPUT_FILE_STREAMS_FSTREAMS_TEST_HELPERS_H

#include <cassert>
#include <concepts>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <type_traits>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <io.h>
#  include <windows.h>
#else
#  include <fcntl.h>
#endif

#include "check_assertion.h"
#include "platform_support.h"
#include "types.h"

inline bool is_handle_valid(NativeHandleT handle) {
#if defined(_WIN32)
  BY_HANDLE_FILE_INFORMATION fileInformation;
  return GetFileInformationByHandle(handle, &fileInformation));
#elif __has_include(<unistd.h>) // POSIX
  return fcntl(handle, F_GETFL) != -1 || errno != EBADF;
#else
#  error "Provide a native file handle!"
#endif
}

template <typename CharT, typename StreamT>
void test_native_handle() {
  static_assert(
      std::is_same_v<typename std::basic_filebuf<CharT>::native_handle_type, typename StreamT::native_handle_type>);

  std::filesystem::path p = get_temp_file_name();

  // non-const
  {
    StreamT f;

    assert(f.open(p) != nullptr);
    assert(f.native_handle() == f.rdbuf()->native_handle());
    std::same_as<NativeHandleT> decltype(auto) handle = f.native_handle();
    assert(is_handle_valid(handle));
    f.close();
    assert(is_handle_valid(handle));
    static_assert(noexcept(f.native_handle()));
  }
  // const
  {
    StreamT cf;

    assert(cf.open(p) != nullptr);
    std::same_as<NativeHandleT> decltype(auto) const_handle = cf.native_handle();
    assert(is_handle_valid(const_handle));
    cf.close();
    assert(!is_handle_valid(const_handle));
    static_assert(noexcept(cf.native_handle()));
  }
}

template <typename StreamT>
void test_native_handle_assertion() {
  std::filesystem::path p = get_temp_file_name();

  // non-const
  {
    StreamT f;

    TEST_LIBCPP_ASSERT_FAILURE(f.native_handle(), "File must be opened");
  }
  // const
  {
    StreamT cf;

    TEST_LIBCPP_ASSERT_FAILURE(cf.native_handle(), "File must be opened");
  }
}

#endif // TEST_STD_INPUT_OUTPUT_FILE_STREAMS_FSTREAMS_TEST_HELPERS_H
