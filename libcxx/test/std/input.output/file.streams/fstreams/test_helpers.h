//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_INPUT_OUTPUT_FILE_STREAMS_FSTREAMS_TEST_HELPERS_H
#define TEST_STD_INPUT_OUTPUT_FILE_STREAMS_FSTREAMS_TEST_HELPERS_H

#include <cstdio>

#if defined(_LIBCPP_WIN32API)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <io.h>
#  include <windows.h>
#else
#  include <fcntl.h>
#endif

#if defined(_LIBCPP_WIN32API)
auto is_handle_valid([[maybe_unused]] HANDLE handle) {
  // TODO: GetFileInformationByHandle
  return true;
};
#else
// POSIX
auto is_handle_valid(int fd) { return fcntl(fd, F_GETFL) != -1 || errno != EBADF; };
#endif

#endif // TEST_STD_INPUT_OUTPUT_FILE_STREAMS_FSTREAMS_TEST_HELPERS_H