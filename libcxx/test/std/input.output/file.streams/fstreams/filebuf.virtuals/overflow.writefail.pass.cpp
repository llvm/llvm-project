//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-filesystem

// setrlimit(RLIMIT_FSIZE) seems to only work as intended on Apple platforms
// REQUIRES: target={{.+}}-apple-{{.+}}

// <fstream>

// Make sure that we properly handle the case where we try to write content to a file
// but we fail to do so because std::fwrite fails.

#include <cassert>
#include <csignal>
#include <cstddef>
#include <fstream>
#include <string>

#include "platform_support.h"
#include "test_macros.h"

#if __has_include(<sys/resource.h>)
#  include <sys/resource.h>
void limit_file_size_to(std::size_t bytes) {
  rlimit lim = {bytes, bytes};
  assert(setrlimit(RLIMIT_FSIZE, &lim) == 0);

  std::signal(SIGXFSZ, [](int) {}); // ignore SIGXFSZ to ensure std::fwrite fails
}
#else
#  error No known way to limit the amount of filesystem space available
#endif

template <class CharT>
void test() {
  std::string temp = get_temp_file_name();
  std::basic_filebuf<CharT> fbuf;
  assert(fbuf.open(temp, std::ios::out | std::ios::trunc));

  std::size_t const limit = 100000;
  limit_file_size_to(limit);

  std::basic_string<CharT> large_block(limit / 10, CharT(42));

  std::streamsize ret;
  std::size_t bytes_written = 0;
  while ((ret = fbuf.sputn(large_block.data(), large_block.size())) != 0) {
    bytes_written += ret;

    // In theory, it's possible for an implementation to allow writing arbitrarily more bytes than
    // set by setrlimit, but in practice if we bust 100x our limit, something else is wrong with the
    // test and we'd end up looping forever.
    assert(bytes_written < 100 * limit);
  }

  fbuf.close();
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
