//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <format>

// basic_format_arg<Context> get(size_t i) const noexcept;

#include <format>
#include <cassert>
#include <limits>
#include <type_traits>

#include "test_macros.h"
#include "make_string.h"

template <class Context, class To, class From>
void test(From value) {
  auto store = std::make_format_args<Context>(value);
  const std::basic_format_args<Context> format_args{store};

  auto visitor = [v = To(value)](auto a) {
    if constexpr (std::is_same_v<To, decltype(a)>)
      assert(v == a);
    else
      assert(false);
  };
#if TEST_STD_VER >= 26 && defined(TEST_HAS_EXPLICIT_THIS_PARAMETER)
  format_args.get(0).visit(visitor);
#else
  std::visit_format_arg(visitor, format_args.get(0));
#endif
}

// Some types, as an extension, are stored in the variant. The Standard
// requires them to be observed as a handle.
template <class Context, class T>
void test_handle(T value) {
  auto store = std::make_format_args<Context>(value);
  std::basic_format_args<Context> format_args{store};

  auto visitor = [](auto a) { assert((std::is_same_v<decltype(a), typename std::basic_format_arg<Context>::handle>)); };
#if TEST_STD_VER >= 26 && defined(TEST_HAS_EXPLICIT_THIS_PARAMETER)
  format_args.get(0).visit(visitor);
#else
  std::visit_format_arg(visitor, format_args.get(0));
#endif
}

// Test specific for string and string_view.
//
// Since both result in a string_view there's no need to pass this as a
// template argument.
template <class Context, class From>
void test_string_view(From value) {
  auto store = std::make_format_args<Context>(value);
  const std::basic_format_args<Context> format_args{store};

  using CharT = typename Context::char_type;
  using To = std::basic_string_view<CharT>;
  using V = std::basic_string<CharT>;

  auto visitor = [v = V(value.begin(), value.end())](auto a) {
    if constexpr (std::is_same_v<To, decltype(a)>)
      assert(v == a);
    else
      assert(false);
  };
#if TEST_STD_VER >= 26 && defined(TEST_HAS_EXPLICIT_THIS_PARAMETER)
  format_args.get(0).visit(visitor);
#else
  std::visit_format_arg(visitor, format_args.get(0));
#endif
}

template <class CharT>
void test() {
  using Context = std::basic_format_context<CharT*, CharT>;
  {
    const std::basic_format_args<Context> format_args{};
    ASSERT_NOEXCEPT(format_args.get(0));
    assert(!format_args.get(0));
  }

  using char_type = typename Context::char_type;
  std::basic_string<char_type> empty;
  std::basic_string<char_type> str = MAKE_STRING(char_type, "abc");

  // Test boolean types.

  test<Context, bool>(true);
  test<Context, bool>(false);

  // Test char_type types.

  test<Context, char_type, char_type>('a');
  test<Context, char_type, char_type>('z');
  test<Context, char_type, char_type>('0');
  test<Context, char_type, char_type>('9');

  // Test char types.

  if (std::is_same_v<char_type, char>) {
    // char to char -> char
    test<Context, char_type, char>('a');
    test<Context, char_type, char>('z');
    test<Context, char_type, char>('0');
    test<Context, char_type, char>('9');
  } else {
    if (std::is_same_v<char_type, wchar_t>) {
      // char to wchar_t -> wchar_t
      test<Context, wchar_t, char>('a');
      test<Context, wchar_t, char>('z');
      test<Context, wchar_t, char>('0');
      test<Context, wchar_t, char>('9');
    } else if (std::is_signed_v<char>) {
      // char to char_type -> int
      // This happens when Context::char_type is a char8_t, char16_t, or
      // char32_t and char is a signed type.
      // Note if sizeof(char_type) > sizeof(int) this test fails. If there are
      // platforms where that occurs extra tests need to be added for char32_t
      // testing it against a long long.
      test<Context, int, char>('a');
      test<Context, int, char>('z');
      test<Context, int, char>('0');
      test<Context, int, char>('9');
    } else {
      // char to char_type -> unsigned
      // This happens when Context::char_type is a char8_t, char16_t, or
      // char32_t and char is an unsigned type.
      // Note if sizeof(char_type) > sizeof(unsigned) this test fails. If there
      // are platforms where that occurs extra tests need to be added for
      // char32_t testing it against an unsigned long long.
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

  using LongToType =
      std::conditional_t<sizeof(long) == sizeof(int), int, long long>;

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
  test<Context, unsigned, unsigned char>(
      std::numeric_limits<unsigned char>::max());

  test<Context, unsigned, unsigned short>(0);
  test<Context, unsigned, unsigned short>(
      std::numeric_limits<unsigned char>::max());
  test<Context, unsigned, unsigned short>(
      std::numeric_limits<unsigned short>::max());

  test<Context, unsigned, unsigned>(0);
  test<Context, unsigned, unsigned>(std::numeric_limits<unsigned char>::max());
  test<Context, unsigned, unsigned>(std::numeric_limits<unsigned short>::max());
  test<Context, unsigned, unsigned>(std::numeric_limits<unsigned>::max());

  using UnsignedLongToType =
      std::conditional_t<sizeof(unsigned long) == sizeof(unsigned), unsigned,
                         unsigned long long>;

  test<Context, UnsignedLongToType, unsigned long>(0);
  test<Context, UnsignedLongToType, unsigned long>(
      std::numeric_limits<unsigned char>::max());
  test<Context, UnsignedLongToType, unsigned long>(
      std::numeric_limits<unsigned short>::max());
  test<Context, UnsignedLongToType, unsigned long>(
      std::numeric_limits<unsigned>::max());
  test<Context, UnsignedLongToType, unsigned long>(
      std::numeric_limits<unsigned long>::max());

  test<Context, unsigned long long, unsigned long long>(0);
  test<Context, unsigned long long, unsigned long long>(
      std::numeric_limits<unsigned char>::max());
  test<Context, unsigned long long, unsigned long long>(
      std::numeric_limits<unsigned short>::max());
  test<Context, unsigned long long, unsigned long long>(
      std::numeric_limits<unsigned>::max());
  test<Context, unsigned long long, unsigned long long>(
      std::numeric_limits<unsigned long>::max());
  test<Context, unsigned long long, unsigned long long>(
      std::numeric_limits<unsigned long long>::max());

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

  test<Context, long double, long double>(
      -std::numeric_limits<long double>::max());
  test<Context, long double, long double>(
      -std::numeric_limits<long double>::min());
  test<Context, long double, long double>(-0.0);
  test<Context, long double, long double>(0.0);
  test<Context, long double, long double>(
      std::numeric_limits<long double>::min());
  test<Context, long double, long double>(
      std::numeric_limits<long double>::max());

  // Test const char_type pointer types.

  test<Context, const char_type*, const char_type*>(empty.c_str());
  test<Context, const char_type*, const char_type*>(str.c_str());

  // Test string_view types.

  test<Context, std::basic_string_view<char_type>>(
      std::basic_string_view<char_type>());
  test<Context, std::basic_string_view<char_type>,
       std::basic_string_view<char_type>>(empty);
  test<Context, std::basic_string_view<char_type>,
       std::basic_string_view<char_type>>(str);

  // Test string types.

  test<Context, std::basic_string_view<char_type>>(
      std::basic_string<char_type>());
  test<Context, std::basic_string_view<char_type>,
       std::basic_string<char_type>>(empty);
  test<Context, std::basic_string_view<char_type>,
       std::basic_string<char_type>>(str);

  // Test pointer types.

  test<Context, const void*>(nullptr);
}

void test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
}

int main(int, char**) {
  test();

  return 0;
}
