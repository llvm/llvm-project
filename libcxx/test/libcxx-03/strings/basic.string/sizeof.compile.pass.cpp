//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that we never change the size or alignment of `basic_string`

#include <cstdint>
#include <iterator>
#include <string>

#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"

template <class T>
class small_pointer {
public:
  using value_type        = T;
  using difference_type   = std::int16_t;
  using pointer           = small_pointer;
  using reference         = T&;
  using iterator_category = std::random_access_iterator_tag;

private:
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
using min_string = std::basic_string<CharT, std::char_traits<CharT>, min_allocator<CharT> >;

template <class CharT>
using test_string = std::basic_string<CharT, std::char_traits<CharT>, test_allocator<CharT> >;

template <class CharT>
using small_string = std::basic_string<CharT, std::char_traits<CharT>, small_iter_allocator<CharT> >;

#if __SIZE_WIDTH__ == 64

static_assert(sizeof(std::string) == 24, "");
static_assert(sizeof(min_string<char>) == 24, "");
static_assert(sizeof(test_string<char>) == 32, "");
static_assert(sizeof(small_string<char>) == 6, "");

#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
#    if __WCHAR_WIDTH__ == 32
static_assert(sizeof(std::wstring) == 24, "");
static_assert(sizeof(min_string<wchar_t>) == 24, "");
static_assert(sizeof(test_string<wchar_t>) == 32, "");
static_assert(sizeof(small_string<wchar_t>) == 12, "");
#    elif __WCHAR_WIDTH__ == 16
static_assert(sizeof(std::wstring) == 24, "");
static_assert(sizeof(min_string<wchar_t>) == 24, "");
static_assert(sizeof(test_string<wchar_t>) == 32, "");
static_assert(sizeof(small_string<wchar_t>) == 6, "");
#    else
#      error "Unexpected wchar_t width"
#    endif
#  endif

#  ifndef TEST_HAS_NO_CHAR8_T
static_assert(sizeof(std::u8string) == 24, "");
static_assert(sizeof(min_string<char8_t>) == 24, "");
static_assert(sizeof(test_string<char8_t>) == 32, "");
static_assert(sizeof(small_string<char8_t>) == 6, "");
#  endif

#  ifndef TEST_HAS_NO_UNICODE_CHARS
static_assert(sizeof(std::u16string) == 24, "");
static_assert(sizeof(std::u32string) == 24, "");
static_assert(sizeof(min_string<char16_t>) == 24, "");
static_assert(sizeof(min_string<char32_t>) == 24, "");
static_assert(sizeof(test_string<char16_t>) == 32, "");
static_assert(sizeof(test_string<char32_t>) == 32, "");
static_assert(sizeof(small_string<char16_t>) == 6, "");
static_assert(sizeof(small_string<char32_t>) == 12, "");
#  endif

#elif __SIZE_WIDTH__ == 32

static_assert(sizeof(std::string) == 12, "");
static_assert(sizeof(min_string<char>) == 12, "");
static_assert(sizeof(test_string<char>) == 24, "");
static_assert(sizeof(small_string<char>) == 6, "");

#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
#    if __WCHAR_WIDTH__ == 32
static_assert(sizeof(std::wstring) == 12, "");
static_assert(sizeof(min_string<wchar_t>) == 12, "");
static_assert(sizeof(test_string<wchar_t>) == 24, "");
static_assert(sizeof(small_string<wchar_t>) == 12, "");
#    elif __WCHAR_WIDTH__ == 16
static_assert(sizeof(std::wstring) == 12, "");
static_assert(sizeof(min_string<wchar_t>) == 12, "");
static_assert(sizeof(test_string<wchar_t>) == 24, "");
static_assert(sizeof(small_string<wchar_t>) == 6, "");
#    else
#      error "Unexpected wchar_t width"
#    endif
#  endif

#  ifndef TEST_HAS_NO_CHAR8_T
static_assert(sizeof(std::u8string) == 12, "");
static_assert(sizeof(min_string<char8_t>) == 12, "");
static_assert(sizeof(test_string<char8_t>) == 24, "");
static_assert(sizeof(small_string<char>) == 6, "");
#  endif

#  ifndef TEST_HAS_NO_UNICODE_CHARS
static_assert(sizeof(std::u16string) == 12, "");
static_assert(sizeof(std::u32string) == 12, "");
static_assert(sizeof(min_string<char16_t>) == 12, "");
static_assert(sizeof(min_string<char32_t>) == 12, "");
static_assert(sizeof(test_string<char16_t>) == 24, "");
static_assert(sizeof(test_string<char32_t>) == 24, "");
static_assert(sizeof(small_string<char16_t>) == 6, "");
static_assert(sizeof(small_string<char32_t>) == 12, "");
#  endif

#else
#  error "std::size_t has an unexpected size"
#endif
