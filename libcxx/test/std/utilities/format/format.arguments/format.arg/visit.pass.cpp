//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME
// The tested functionality needs deducing this.
// XFAIL: apple-clang

// <format>

// class basic_format_arg;

// template<class Visitor>
//   decltype(auto) visit(this basic_format_arg arg, Visitor&& vis);  // since C++26

#include <algorithm>
#include <cassert>
#include <format>
#include <type_traits>

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_macros.h"

template <class Context, class To, class From>
void test(From value) {
  auto store = std::make_format_args<Context>(value);
  std::basic_format_args<Context> format_args{store};

  LIBCPP_ASSERT(format_args.__size() == 1);
  assert(format_args.get(0));

  auto result = format_args.get(0).visit([v = To(value)](auto a) -> To {
    if constexpr (std::is_same_v<To, decltype(a)>) {
      assert(v == a);
      return a;
    } else {
      assert(false);
      return {};
    }
  });

  using ct = std::common_type_t<From, To>;
  assert(static_cast<ct>(result) == static_cast<ct>(value));
}

// Some types, as an extension, are stored in the variant. The Standard
// requires them to be observed as a handle.
template <class Context, class T>
void test_handle(T value) {
  auto store = std::make_format_args<Context>(value);
  std::basic_format_args<Context> format_args{store};

  LIBCPP_ASSERT(format_args.__size() == 1);
  assert(format_args.get(0));

  format_args.get(0).visit([](auto a) {
    assert((std::is_same_v<decltype(a), typename std::basic_format_arg<Context>::handle>));
  });
}

// Test specific for string and string_view.
//
// Since both result in a string_view there's no need to pass this as a
// template argument.
template <class Context, class From>
void test_string_view(From value) {
  auto store = std::make_format_args<Context>(value);
  std::basic_format_args<Context> format_args{store};

  LIBCPP_ASSERT(format_args.__size() == 1);
  assert(format_args.get(0));

  using CharT = typename Context::char_type;
  using To    = std::basic_string_view<CharT>;
  using V     = std::basic_string<CharT>;

  auto result = format_args.get(0).visit([v = V(value.begin(), value.end())](auto a) -> To {
    if constexpr (std::is_same_v<To, decltype(a)>) {
      assert(v == a);
      return a;
    } else {
      assert(false);
      return {};
    }
  });

  assert(std::equal(value.begin(), value.end(), result.begin(), result.end()));
}

template <class CharT>
void test() {
  using Context = std::basic_format_context<CharT*, CharT>;
  std::basic_string<CharT> empty;
  std::basic_string<CharT> str = MAKE_STRING(CharT, "abc");

  // Test boolean types.

  test<Context, bool>(true);
  test<Context, bool>(false);

  // Test CharT types.

  test<Context, CharT, CharT>('a');
  test<Context, CharT, CharT>('z');
  test<Context, CharT, CharT>('0');
  test<Context, CharT, CharT>('9');

  // Test char types.

  if (std::is_same_v<CharT, char>) {
    // char to char -> char
    test<Context, CharT, char>('a');
    test<Context, CharT, char>('z');
    test<Context, CharT, char>('0');
    test<Context, CharT, char>('9');
  } else {
    if (std::is_same_v<CharT, wchar_t>) {
      // char to wchar_t -> wchar_t
      test<Context, wchar_t, char>('a');
      test<Context, wchar_t, char>('z');
      test<Context, wchar_t, char>('0');
      test<Context, wchar_t, char>('9');
    } else if (std::is_signed_v<char>) {
      // char to CharT -> int
      // This happens when CharT is a char8_t, char16_t, or char32_t and char
      // is a signed type.
      // Note if sizeof(CharT) > sizeof(int) this test fails. If there are
      // platforms where that occurs extra tests need to be added for char32_t
      // testing it against a long long.
      test<Context, int, char>('a');
      test<Context, int, char>('z');
      test<Context, int, char>('0');
      test<Context, int, char>('9');
    } else {
      // char to CharT -> unsigned
      // This happens when CharT is a char8_t, char16_t, or char32_t and char
      // is an unsigned type.
      // Note if sizeof(CharT) > sizeof(unsigned) this test fails. If there are
      // platforms where that occurs extra tests need to be added for char32_t
      // testing it against an unsigned long long.
      test<Context, unsigned, char>('a');
      test<Context, unsigned, char>('z');
      test<Context, unsigned, char>('0');
      test<Context, unsigned, char>('9');
    }
  }

  // Test signed integer types.

  test<Context, int, signed char>(std::numeric_limits<signed char>::min());
  test<Context, int, signed char>(0);
  test<Context, int, signed char>(std::numeric_limits<signed char>::max());

  test<Context, int, short>(std::numeric_limits<short>::min());
  test<Context, int, short>(std::numeric_limits<signed char>::min());
  test<Context, int, short>(0);
  test<Context, int, short>(std::numeric_limits<signed char>::max());
  test<Context, int, short>(std::numeric_limits<short>::max());

  test<Context, int, int>(std::numeric_limits<int>::min());
  test<Context, int, int>(std::numeric_limits<short>::min());
  test<Context, int, int>(std::numeric_limits<signed char>::min());
  test<Context, int, int>(0);
  test<Context, int, int>(std::numeric_limits<signed char>::max());
  test<Context, int, int>(std::numeric_limits<short>::max());
  test<Context, int, int>(std::numeric_limits<int>::max());

  using LongToType = std::conditional_t<sizeof(long) == sizeof(int), int, long long>;

  test<Context, LongToType, long>(std::numeric_limits<long>::min());
  test<Context, LongToType, long>(std::numeric_limits<int>::min());
  test<Context, LongToType, long>(std::numeric_limits<short>::min());
  test<Context, LongToType, long>(std::numeric_limits<signed char>::min());
  test<Context, LongToType, long>(0);
  test<Context, LongToType, long>(std::numeric_limits<signed char>::max());
  test<Context, LongToType, long>(std::numeric_limits<short>::max());
  test<Context, LongToType, long>(std::numeric_limits<int>::max());
  test<Context, LongToType, long>(std::numeric_limits<long>::max());

  test<Context, long long, long long>(std::numeric_limits<long long>::min());
  test<Context, long long, long long>(std::numeric_limits<long>::min());
  test<Context, long long, long long>(std::numeric_limits<int>::min());
  test<Context, long long, long long>(std::numeric_limits<short>::min());
  test<Context, long long, long long>(std::numeric_limits<signed char>::min());
  test<Context, long long, long long>(0);
  test<Context, long long, long long>(std::numeric_limits<signed char>::max());
  test<Context, long long, long long>(std::numeric_limits<short>::max());
  test<Context, long long, long long>(std::numeric_limits<int>::max());
  test<Context, long long, long long>(std::numeric_limits<long>::max());
  test<Context, long long, long long>(std::numeric_limits<long long>::max());

#ifndef TEST_HAS_NO_INT128
  test_handle<Context, __int128_t>(0);
#endif // TEST_HAS_NO_INT128

  // Test unsigned integer types.

  test<Context, unsigned, unsigned char>(0);
  test<Context, unsigned, unsigned char>(std::numeric_limits<unsigned char>::max());

  test<Context, unsigned, unsigned short>(0);
  test<Context, unsigned, unsigned short>(std::numeric_limits<unsigned char>::max());
  test<Context, unsigned, unsigned short>(std::numeric_limits<unsigned short>::max());

  test<Context, unsigned, unsigned>(0);
  test<Context, unsigned, unsigned>(std::numeric_limits<unsigned char>::max());
  test<Context, unsigned, unsigned>(std::numeric_limits<unsigned short>::max());
  test<Context, unsigned, unsigned>(std::numeric_limits<unsigned>::max());

  using UnsignedLongToType =
      std::conditional_t<sizeof(unsigned long) == sizeof(unsigned), unsigned, unsigned long long>;

  test<Context, UnsignedLongToType, unsigned long>(0);
  test<Context, UnsignedLongToType, unsigned long>(std::numeric_limits<unsigned char>::max());
  test<Context, UnsignedLongToType, unsigned long>(std::numeric_limits<unsigned short>::max());
  test<Context, UnsignedLongToType, unsigned long>(std::numeric_limits<unsigned>::max());
  test<Context, UnsignedLongToType, unsigned long>(std::numeric_limits<unsigned long>::max());

  test<Context, unsigned long long, unsigned long long>(0);
  test<Context, unsigned long long, unsigned long long>(std::numeric_limits<unsigned char>::max());
  test<Context, unsigned long long, unsigned long long>(std::numeric_limits<unsigned short>::max());
  test<Context, unsigned long long, unsigned long long>(std::numeric_limits<unsigned>::max());
  test<Context, unsigned long long, unsigned long long>(std::numeric_limits<unsigned long>::max());
  test<Context, unsigned long long, unsigned long long>(std::numeric_limits<unsigned long long>::max());

#ifndef TEST_HAS_NO_INT128
  test_handle<Context, __uint128_t>(0);
#endif // TEST_HAS_NO_INT128

  // Test floating point types.

  test<Context, float, float>(-std::numeric_limits<float>::max());
  test<Context, float, float>(-std::numeric_limits<float>::min());
  test<Context, float, float>(-0.0);
  test<Context, float, float>(0.0);
  test<Context, float, float>(std::numeric_limits<float>::min());
  test<Context, float, float>(std::numeric_limits<float>::max());

  test<Context, double, double>(-std::numeric_limits<double>::max());
  test<Context, double, double>(-std::numeric_limits<double>::min());
  test<Context, double, double>(-0.0);
  test<Context, double, double>(0.0);
  test<Context, double, double>(std::numeric_limits<double>::min());
  test<Context, double, double>(std::numeric_limits<double>::max());

  test<Context, long double, long double>(-std::numeric_limits<long double>::max());
  test<Context, long double, long double>(-std::numeric_limits<long double>::min());
  test<Context, long double, long double>(-0.0);
  test<Context, long double, long double>(0.0);
  test<Context, long double, long double>(std::numeric_limits<long double>::min());
  test<Context, long double, long double>(std::numeric_limits<long double>::max());

  // Test const CharT pointer types.

  test<Context, const CharT*, const CharT*>(empty.c_str());
  test<Context, const CharT*, const CharT*>(str.c_str());

  // Test string_view types.

  {
    using From = std::basic_string_view<CharT>;

    test_string_view<Context>(From());
    test_string_view<Context>(From(empty.c_str()));
    test_string_view<Context>(From(str.c_str()));
  }

  {
    using From = std::basic_string_view<CharT, constexpr_char_traits<CharT>>;

    test_string_view<Context>(From());
    test_string_view<Context>(From(empty.c_str()));
    test_string_view<Context>(From(str.c_str()));
  }

  // Test string types.

  {
    using From = std::basic_string<CharT>;

    test_string_view<Context>(From());
    test_string_view<Context>(From(empty.c_str()));
    test_string_view<Context>(From(str.c_str()));
  }

  {
    using From = std::basic_string<CharT, constexpr_char_traits<CharT>, std::allocator<CharT>>;

    test_string_view<Context>(From());
    test_string_view<Context>(From(empty.c_str()));
    test_string_view<Context>(From(str.c_str()));
  }

  {
    using From = std::basic_string<CharT, std::char_traits<CharT>, min_allocator<CharT>>;

    test_string_view<Context>(From());
    test_string_view<Context>(From(empty.c_str()));
    test_string_view<Context>(From(str.c_str()));
  }

  {
    using From = std::basic_string<CharT, constexpr_char_traits<CharT>, min_allocator<CharT>>;

    test_string_view<Context>(From());
    test_string_view<Context>(From(empty.c_str()));
    test_string_view<Context>(From(str.c_str()));
  }

  // Test pointer types.

  test<Context, const void*>(nullptr);
  int i = 0;
  test<Context, const void*>(static_cast<void*>(&i));
  const int ci = 0;
  test<Context, const void*>(static_cast<const void*>(&ci));
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
