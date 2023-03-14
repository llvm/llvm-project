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
static const std::size_t alignment = 16;

template <class = int>
TEST_CONSTEXPR_CXX20 void full_size() {
  std::string str;
  assert(str.max_size() == std::numeric_limits<std::size_t>::max() - alignment);

#ifndef TEST_HAS_NO_CHAR8_T
  std::u8string u8str;
  assert(u8str.max_size() == std::numeric_limits<std::size_t>::max() - alignment);
#endif

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::wstring wstr;
  assert(wstr.max_size() == std::numeric_limits<std::size_t>::max() / sizeof(wchar_t) - alignment);
#endif

  std::u16string u16str;
  std::u32string u32str;
  assert(u16str.max_size() == std::numeric_limits<std::size_t>::max() / 2 - alignment);
  assert(u32str.max_size() == std::numeric_limits<std::size_t>::max() / 4 - alignment);
}

template <class = int>
TEST_CONSTEXPR_CXX20 void half_size() {
  std::string str;
  assert(str.max_size() == std::numeric_limits<std::size_t>::max() / 2 - alignment);

#ifndef TEST_HAS_NO_CHAR8_T
  std::u8string u8str;
  assert(u8str.max_size() == std::numeric_limits<std::size_t>::max() / 2 - alignment);
#endif

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  std::wstring wstr;
  assert(wstr.max_size() == std::numeric_limits<std::size_t>::max() / std::max<size_t>(2ul, sizeof(wchar_t)) - alignment);
#endif

  std::u16string u16str;
  std::u32string u32str;
  assert(u16str.max_size() == std::numeric_limits<std::size_t>::max() / 2 - alignment);
  assert(u32str.max_size() == std::numeric_limits<std::size_t>::max() / 4 - alignment);
}

TEST_CONSTEXPR_CXX20 bool test() {

#if _LIBCPP_ABI_VERSION == 1

# if defined(__x86_64__) || defined(__i386__)
  full_size();
# elif defined(__APPLE__) && defined(__aarch64__)
  half_size();
# elif defined(__arm__) || defined(__aarch64__)
#   ifdef __BIG_ENDIAN__
  half_size();
#   else
  full_size();
#   endif
# elif defined(__powerpc__) || defined(__powerpc64__)
#   ifdef __BIG_ENDIAN__
  half_size();
#   else
  full_size();
#   endif
# elif defined(__sparc64__)
  half_size();
# elif defined(_WIN32)
  full_size();
# else
#   error "Your target system seems to be unsupported."
# endif

#else

  half_size();

#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
