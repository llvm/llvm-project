//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO(mordante) Investigate
// UNSUPPORTED: apple-clang

// <fstream>

// basic_streambuf<charT, traits>* setbuf(char_type* s, streamsize n) override;

// This test requires the fix to https://llvm.org/PR60509 in the dylib,
// which landed in 5afb937d8a30445642ccaf33866ee4cdd0713222.
// XFAIL: using-built-library-before-llvm-19

#include <fstream>
#include <cstddef>
#include <cassert>

#include "test_macros.h"

template <class CharT>
static std::size_t file_size(const char* filename) {
  FILE* f = std::fopen(filename, "rb");
  std::fseek(f, 0, SEEK_END);
  long result = std::ftell(f);
  std::fclose(f);
  return result;
}

// Helper class to expose some protected std::basic_filebuf<CharT> members.
template <class CharT>
struct filebuf : public std::basic_filebuf<CharT> {
  CharT* base() { return this->pbase(); }
  CharT* ptr() { return this->pptr(); }
};

template <class CharT>
static void buffered_request() {
  filebuf<CharT> buffer;

  CharT b[10] = {0};
  assert(buffer.pubsetbuf(b, 10) == &buffer);

  buffer.open("test.dat", std::ios_base::out);
  buffer.sputc(CharT('a'));
  assert(b[0] == 'a');

  buffer.close();
  assert(file_size<CharT>("test.dat") == 1);
}

template <class CharT>
static void unbuffered_request_before_open() {
  filebuf<CharT> buffer;

  assert(buffer.pubsetbuf(nullptr, 0) == &buffer);
  assert(buffer.base() == nullptr);
  assert(buffer.ptr() == nullptr);

  buffer.open("test.dat", std::ios_base::out);
  assert(buffer.base() == nullptr);
  assert(buffer.ptr() == nullptr);

  buffer.sputc(CharT('a'));
  assert(buffer.base() == nullptr);
  assert(buffer.ptr() == nullptr);

  assert(file_size<CharT>("test.dat") == 1);
}

template <class CharT>
static void unbuffered_request_after_open() {
  filebuf<CharT> buffer;

  buffer.open("test.dat", std::ios_base::out);

  assert(buffer.pubsetbuf(nullptr, 0) == &buffer);
  assert(buffer.base() == nullptr);
  assert(buffer.ptr() == nullptr);

  buffer.sputc(CharT('a'));
  assert(buffer.base() == nullptr);
  assert(buffer.ptr() == nullptr);

  assert(file_size<CharT>("test.dat") == 1);
}

template <class CharT>
static void unbuffered_request_after_open_ate() {
  filebuf<CharT> buffer;

  buffer.open("test.dat", std::ios_base::out | std::ios_base::ate);

  assert(buffer.pubsetbuf(nullptr, 0) == &buffer);

  buffer.sputc(CharT('a'));
  assert(file_size<CharT>("test.dat") <= 1);
  // on libc++ buffering is used by default.
  LIBCPP_ASSERT(file_size<CharT>("test.dat") == 0);

  buffer.close();
  assert(file_size<CharT>("test.dat") == 1);
}

template <class CharT>
static void test() {
  buffered_request<CharT>();

  unbuffered_request_before_open<CharT>();
  unbuffered_request_after_open<CharT>();
  unbuffered_request_after_open_ate<CharT>();
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
