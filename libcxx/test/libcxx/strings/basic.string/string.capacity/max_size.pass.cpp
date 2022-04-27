//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// This test ensures that the correct max_size() is returned depending on the platform.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>

#include "test_macros.h"

// alignment of the string heap buffer is hardcoded to 16
static const size_t alignment = 16;

void full_size() {
  std::string str;
  assert(str.max_size() == std::numeric_limits<size_t>::max() - alignment);

#ifndef TEST_HAS_NO_CHAR8_T
  std::u8string u8str;
  assert(u8str.max_size() == std::numeric_limits<size_t>::max() - alignment);
#endif

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::wstring wstr;
  assert(wstr.max_size() == std::numeric_limits<size_t>::max() / sizeof(wchar_t) - alignment);
#endif

#ifndef TEST_HAS_NO_UNICODE_CHARS
  std::u16string u16str;
  std::u32string u32str;
  assert(u16str.max_size() == std::numeric_limits<size_t>::max() / 2 - alignment);
  assert(u32str.max_size() == std::numeric_limits<size_t>::max() / 4 - alignment);
#endif
}

void half_size() {
  std::string str;
  assert(str.max_size() == std::numeric_limits<size_t>::max() / 2 - alignment);

#ifndef TEST_HAS_NO_CHAR8_T
  std::u8string u8str;
  assert(u8str.max_size() == std::numeric_limits<size_t>::max() / 2 - alignment);
#endif

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::wstring wstr;
  assert(wstr.max_size() == std::numeric_limits<size_t>::max() / std::max<size_t>(2ul, sizeof(wchar_t)) - alignment);
#endif

#ifndef TEST_HAS_NO_UNICODE_CHARS
  std::u16string u16str;
  std::u32string u32str;
  assert(u16str.max_size() == std::numeric_limits<size_t>::max() / 2 - alignment);
  assert(u32str.max_size() == std::numeric_limits<size_t>::max() / 4 - alignment);
#endif
}

bool test() {
#if _LIBCPP_ABI_VERSION == 1
#ifdef __APPLE__

#  ifdef __aarch64__
  half_size();
#  elif defined(__x86_64__)
  full_size();
#  else
#    error "Your target system seems to be unsupported."
#  endif

#elif defined(__linux__)

#  ifdef  __x86_64__
  full_size();
#  elif defined(__arm__) || defined(__aarch64__)
#    ifdef __BIG_ENDIAN__
  half_size();
#    else
  full_size();
#    endif

#  else
#    error "Your target system seems to be unsupported."
#  endif

#elif defined(__powerpc__)
  half_size();
#elif defined(__powepc64__)
  half_size();

#elif defined(_WIN64)

#  ifdef __x86_64__
  full_size();
#  else
#    error "Your target system seems to be unsupported."
#  endif

#else
#  error "Your target system seems to be unsupported."
#endif

#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  // static_assert(test());
#endif

  return 0;
}
