//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that we never change the size or alignment of `basic_string`

// UNSUPPORTED: c++03

#include <string>

#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"

template <class T>
class small_pointer {
  std::uint16_t offset;
};

template <class T>
class small_iter_allocator {
public:
  using value_type      = T;
  using pointer         = small_pointer<T>;
  using size_type       = std::int16_t;
  using difference_type = std::int16_t;

  small_iter_allocator() TEST_NOEXCEPT {}

  template <class U>
  small_iter_allocator(small_iter_allocator<U>) TEST_NOEXCEPT {}

  T* allocate(std::size_t n);
  void deallocate(T* p, std::size_t);

  friend bool operator==(small_iter_allocator, small_iter_allocator) { return true; }
  friend bool operator!=(small_iter_allocator, small_iter_allocator) { return false; }
};

template <class CharT>
using min_string = std::basic_string<CharT, std::char_traits<CharT>, min_allocator<CharT>>;

template <class CharT>
using test_string = std::basic_string<CharT, std::char_traits<CharT>, test_allocator<CharT>>;

template <class CharT>
using small_string = std::basic_string<CharT, std::char_traits<CharT>, small_iter_allocator<CharT>>;

#if __SIZE_WIDTH__ == 64

static_assert(alignof(std::string) == 8, "");
static_assert(alignof(min_string<char>) == 8, "");
static_assert(alignof(test_string<char>) == 8, "");
static_assert(alignof(small_string<char>) == 2, "");

#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
#    if __WCHAR_WIDTH__ == 32
static_assert(alignof(std::wstring) == 8, "");
static_assert(alignof(min_string<wchar_t>) == 8, "");
static_assert(alignof(test_string<wchar_t>) == 8, "");
static_assert(alignof(small_string<wchar_t>) == 4, "");
#    elif __WCHAR_WIDTH__ == 16
static_assert(alignof(std::wstring) == 8, "");
static_assert(alignof(min_string<wchar_t>) == 8, "");
static_assert(alignof(test_string<wchar_t>) == 8, "");
static_assert(alignof(small_string<wchar_t>) == 2, "");
#    else
#      error "Unexpected wchar_t width"
#    endif
#  endif

#  ifndef TEST_HAS_NO_CHAR8_T
static_assert(alignof(std::u8string) == 8, "");
static_assert(alignof(min_string<char8_t>) == 8, "");
static_assert(alignof(test_string<char8_t>) == 8, "");
static_assert(alignof(small_string<char8_t>) == 2, "");
#  endif

#  ifndef TEST_HAS_NO_UNICODE_CHARS
static_assert(alignof(std::u16string) == 8, "");
static_assert(alignof(std::u32string) == 8, "");
static_assert(alignof(min_string<char16_t>) == 8, "");
static_assert(alignof(min_string<char32_t>) == 8, "");
static_assert(alignof(test_string<char16_t>) == 8, "");
static_assert(alignof(test_string<char32_t>) == 8, "");
static_assert(alignof(small_string<char16_t>) == 2, "");
static_assert(alignof(small_string<char32_t>) == 4, "");
#  endif

#elif __SIZE_WIDTH__ == 32

static_assert(alignof(std::string) == 4, "");
static_assert(alignof(min_string<char>) == 4, "");
static_assert(alignof(test_string<char>) == 4, "");
static_assert(alignof(small_string<char>) == 2, "");

#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
#    if __WCHAR_WIDTH__ == 32
static_assert(alignof(std::wstring) == 4, "");
static_assert(alignof(min_string<wchar_t>) == 4, "");
static_assert(alignof(test_string<wchar_t>) == 4, "");
static_assert(alignof(small_string<wchar_t>) == 4, "");
#    elif __WCHAR_WIDTH__ == 16
static_assert(alignof(std::wstring) == 4, "");
static_assert(alignof(min_string<wchar_t>) == 4, "");
static_assert(alignof(test_string<wchar_t>) == 4, "");
static_assert(alignof(small_string<wchar_t>) == 2, "");
#    else
#      error "Unexpected wchar_t width"
#    endif
#  endif

#  ifndef TEST_HAS_NO_CHAR8_T
static_assert(alignof(std::u8string) == 4, "");
static_assert(alignof(min_string<char8_t>) == 4, "");
static_assert(alignof(test_string<char8_t>) == 4, "");
static_assert(alignof(small_string<char8_t>) == 2, "");
#  endif

#  ifndef TEST_HAS_NO_UNICODE_CHARS
static_assert(alignof(std::u16string) == 4, "");
static_assert(alignof(std::u32string) == 4, "");
static_assert(alignof(min_string<char16_t>) == 4, "");
static_assert(alignof(min_string<char32_t>) == 4, "");
static_assert(alignof(test_string<char16_t>) == 4, "");
static_assert(alignof(test_string<char32_t>) == 4, "");
static_assert(alignof(small_string<char32_t>) == 4, "");
#  endif

#else
#  error "std::size_t has an unexpected size"
#endif
