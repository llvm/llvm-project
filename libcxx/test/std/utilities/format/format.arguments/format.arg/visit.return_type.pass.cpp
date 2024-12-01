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
// UNSUPPORTED: clang-17
// XFAIL: apple-clang

// <format>

// class basic_format_arg;

// template<class R, class Visitor>
//   R visit(this basic_format_arg arg, Visitor&& vis);

#include <algorithm>
#include <cassert>
#include <format>
#include <type_traits>

#include "constexpr_char_traits.h"
#include "make_string.h"
#include "min_allocator.h"
#include "test_macros.h"

// The expected result type shouldn't matter,therefore use a hardcoded value for simplicity.
using ExpectedResultType = bool;
constexpr ExpectedResultType visited{true};

template <class ExpectedR>
ExpectedR make_expected_result() {
  if constexpr (std::is_same_v<ExpectedR, bool>) {
    return true;
  } else if constexpr (std::is_same_v<ExpectedR, long>) {
    return 192812079084L;
  } else {
    return "visited";
  }
}

template <class Context, class To, class ExpectedR, class From>
void test(From value, const ExpectedR& expectedValue) {
  auto store = std::make_format_args<Context>(value);
  std::basic_format_args<Context> format_args{store};

  LIBCPP_ASSERT(format_args.__size() == 1);
  assert(format_args.get(0));

  // member
  {
    std::same_as<ExpectedR> decltype(auto) result =
        format_args.get(0).template visit<ExpectedR>([v = To(value)](auto a) -> ExpectedR {
          if constexpr (std::is_same_v<To, decltype(a)>) {
            assert(v == a);
            return make_expected_result<ExpectedR>();
          } else {
            assert(false);
            return {};
          }
        });

    assert(result == expectedValue);
  }
}

// Some types, as an extension, are stored in the variant. The Standard
// requires them to be observed as a handle.
template <class Context, class T, class ExpectedR>
void test_handle(T value, ExpectedR expectedValue) {
  auto store = std::make_format_args<Context>(value);
  std::basic_format_args<Context> format_args{store};

  LIBCPP_ASSERT(format_args.__size() == 1);
  assert(format_args.get(0));

  std::same_as<ExpectedR> decltype(auto) result = format_args.get(0).template visit<ExpectedR>([](auto a) -> ExpectedR {
    assert((std::is_same_v<decltype(a), typename std::basic_format_arg<Context>::handle>));

    return make_expected_result<ExpectedR>();
  });

  assert(result == expectedValue);
}

// Test specific for string and string_view.
//
// Since both result in a string_view there's no need to pass this as a
// template argument.
template <class Context, class ExpectedR, class From>
void test_string_view(From value, ExpectedR expectedValue) {
  auto store = std::make_format_args<Context>(value);
  std::basic_format_args<Context> format_args{store};

  LIBCPP_ASSERT(format_args.__size() == 1);
  assert(format_args.get(0));

  using CharT = typename Context::char_type;
  using To    = std::basic_string_view<CharT>;
  using V     = std::basic_string<CharT>;

  std::same_as<ExpectedR> decltype(auto) result =
      format_args.get(0).template visit<ExpectedR>([v = V(value.begin(), value.end())](auto a) -> ExpectedR {
        if constexpr (std::is_same_v<To, decltype(a)>) {
          assert(v == a);
          return make_expected_result<ExpectedR>();
        } else {
          assert(false);
          return {};
        }
      });

  assert(result == expectedValue);
}

template <class CharT>
void test() {
  using Context = std::basic_format_context<CharT*, CharT>;
  std::basic_string<CharT> empty;
  std::basic_string<CharT> str = MAKE_STRING(CharT, "abc");

  // Test boolean types.

  test<Context, bool, ExpectedResultType>(true, visited);
  test<Context, bool, ExpectedResultType>(false, visited);

  test<Context, bool, std::string>(true, "visited");
  test<Context, bool, std::string>(false, "visited");

  // Test CharT types.

  test<Context, CharT, ExpectedResultType, CharT>('a', visited);
  test<Context, CharT, ExpectedResultType, CharT>('z', visited);
  test<Context, CharT, ExpectedResultType, CharT>('0', visited);
  test<Context, CharT, ExpectedResultType, CharT>('9', visited);

  // Test char types.

  if (std::is_same_v<CharT, char>) {
    // char to char -> char
    test<Context, CharT, ExpectedResultType, char>('a', visited);
    test<Context, CharT, ExpectedResultType, char>('z', visited);
    test<Context, CharT, ExpectedResultType, char>('0', visited);
    test<Context, CharT, ExpectedResultType, char>('9', visited);
  } else {
    if (std::is_same_v<CharT, wchar_t>) {
      // char to wchar_t -> wchar_t
      test<Context, wchar_t, ExpectedResultType, char>('a', visited);
      test<Context, wchar_t, ExpectedResultType, char>('z', visited);
      test<Context, wchar_t, ExpectedResultType, char>('0', visited);
      test<Context, wchar_t, ExpectedResultType, char>('9', visited);
    } else if (std::is_signed_v<char>) {
      // char to CharT -> int
      // This happens when CharT is a char8_t, char16_t, or char32_t and char
      // is a signed type.
      // Note if sizeof(CharT) > sizeof(int) this test fails. If there are
      // platforms where that occurs extra tests need to be added for char32_t
      // testing it against a long long.
      test<Context, int, ExpectedResultType, char>('a', visited);
      test<Context, int, ExpectedResultType, char>('z', visited);
      test<Context, int, ExpectedResultType, char>('0', visited);
      test<Context, int, ExpectedResultType, char>('9', visited);
    } else {
      // char to CharT -> unsigned
      // This happens when CharT is a char8_t, char16_t, or char32_t and char
      // is an unsigned type.
      // Note if sizeof(CharT) > sizeof(unsigned) this test fails. If there are
      // platforms where that occurs extra tests need to be added for char32_t
      // testing it against an unsigned long long.
      test<Context, unsigned, ExpectedResultType, char>('a', visited);
      test<Context, unsigned, ExpectedResultType, char>('z', visited);
      test<Context, unsigned, ExpectedResultType, char>('0', visited);
      test<Context, unsigned, ExpectedResultType, char>('9', visited);
    }
  }

  // Test signed integer types.

  test<Context, int, ExpectedResultType, signed char>(std::numeric_limits<signed char>::min(), visited);
  test<Context, int, ExpectedResultType, signed char>(0, visited);
  test<Context, int, ExpectedResultType, signed char>(std::numeric_limits<signed char>::max(), visited);

  test<Context, int, ExpectedResultType, short>(std::numeric_limits<short>::min(), visited);
  test<Context, int, ExpectedResultType, short>(std::numeric_limits<signed char>::min(), visited);
  test<Context, int, ExpectedResultType, short>(0, visited);
  test<Context, int, ExpectedResultType, short>(std::numeric_limits<signed char>::max(), visited);
  test<Context, int, ExpectedResultType, short>(std::numeric_limits<short>::max(), visited);

  test<Context, int, ExpectedResultType, int>(std::numeric_limits<int>::min(), visited);
  test<Context, int, ExpectedResultType, int>(std::numeric_limits<short>::min(), visited);
  test<Context, int, ExpectedResultType, int>(std::numeric_limits<signed char>::min(), visited);
  test<Context, int, ExpectedResultType, int>(0, visited);
  test<Context, int, ExpectedResultType, int>(std::numeric_limits<signed char>::max(), visited);
  test<Context, int, ExpectedResultType, int>(std::numeric_limits<short>::max(), visited);
  test<Context, int, ExpectedResultType, int>(std::numeric_limits<int>::max(), visited);

  using LongToType = std::conditional_t<sizeof(long) == sizeof(int), int, long long>;

  test<Context, LongToType, ExpectedResultType, long>(std::numeric_limits<long>::min(), visited);
  test<Context, LongToType, ExpectedResultType, long>(std::numeric_limits<int>::min(), visited);
  test<Context, LongToType, ExpectedResultType, long>(std::numeric_limits<short>::min(), visited);
  test<Context, LongToType, ExpectedResultType, long>(std::numeric_limits<signed char>::min(), visited);
  test<Context, LongToType, ExpectedResultType, long>(0, visited);
  test<Context, LongToType, ExpectedResultType, long>(std::numeric_limits<signed char>::max(), visited);
  test<Context, LongToType, ExpectedResultType, long>(std::numeric_limits<short>::max(), visited);
  test<Context, LongToType, ExpectedResultType, long>(std::numeric_limits<int>::max(), visited);
  test<Context, LongToType, ExpectedResultType, long>(std::numeric_limits<long>::max(), visited);

  test<Context, long long, ExpectedResultType, long long>(std::numeric_limits<long long>::min(), visited);
  test<Context, long long, ExpectedResultType, long long>(std::numeric_limits<long>::min(), visited);
  test<Context, long long, ExpectedResultType, long long>(std::numeric_limits<int>::min(), visited);
  test<Context, long long, ExpectedResultType, long long>(std::numeric_limits<short>::min(), visited);
  test<Context, long long, ExpectedResultType, long long>(std::numeric_limits<signed char>::min(), visited);
  test<Context, long long, ExpectedResultType, long long>(0, visited);
  test<Context, long long, ExpectedResultType, long long>(std::numeric_limits<signed char>::max(), visited);
  test<Context, long long, ExpectedResultType, long long>(std::numeric_limits<short>::max(), visited);
  test<Context, long long, ExpectedResultType, long long>(std::numeric_limits<int>::max(), visited);
  test<Context, long long, ExpectedResultType, long long>(std::numeric_limits<long>::max(), visited);
  test<Context, long long, ExpectedResultType, long long>(std::numeric_limits<long long>::max(), visited);

#ifndef TEST_HAS_NO_INT128
  test_handle<Context, __int128_t, ExpectedResultType>(0, visited);
#endif // TEST_HAS_NO_INT128

  // Test unsigned integer types.

  test<Context, unsigned, ExpectedResultType, unsigned char>(0, visited);
  test<Context, unsigned, ExpectedResultType, unsigned char>(std::numeric_limits<unsigned char>::max(), visited);

  test<Context, unsigned, ExpectedResultType, unsigned short>(0, visited);
  test<Context, unsigned, ExpectedResultType, unsigned short>(std::numeric_limits<unsigned char>::max(), visited);
  test<Context, unsigned, ExpectedResultType, unsigned short>(std::numeric_limits<unsigned short>::max(), visited);

  test<Context, unsigned, ExpectedResultType, unsigned>(0, visited);
  test<Context, unsigned, ExpectedResultType, unsigned>(std::numeric_limits<unsigned char>::max(), visited);
  test<Context, unsigned, ExpectedResultType, unsigned>(std::numeric_limits<unsigned short>::max(), visited);
  test<Context, unsigned, ExpectedResultType, unsigned>(std::numeric_limits<unsigned>::max(), visited);

  using UnsignedLongToType =
      std::conditional_t<sizeof(unsigned long) == sizeof(unsigned), unsigned, unsigned long long>;

  test<Context, UnsignedLongToType, ExpectedResultType, unsigned long>(0, visited);
  test<Context, UnsignedLongToType, ExpectedResultType, unsigned long>(
      std::numeric_limits<unsigned char>::max(), visited);
  test<Context, UnsignedLongToType, ExpectedResultType, unsigned long>(
      std::numeric_limits<unsigned short>::max(), visited);
  test<Context, UnsignedLongToType, ExpectedResultType, unsigned long>(std::numeric_limits<unsigned>::max(), visited);
  test<Context, UnsignedLongToType, ExpectedResultType, unsigned long>(
      std::numeric_limits<unsigned long>::max(), visited);

  test<Context, unsigned long long, ExpectedResultType, unsigned long long>(0, visited);
  test<Context, unsigned long long, ExpectedResultType, unsigned long long>(
      std::numeric_limits<unsigned char>::max(), visited);
  test<Context, unsigned long long, ExpectedResultType, unsigned long long>(
      std::numeric_limits<unsigned short>::max(), visited);
  test<Context, unsigned long long, ExpectedResultType, unsigned long long>(
      std::numeric_limits<unsigned>::max(), visited);
  test<Context, unsigned long long, ExpectedResultType, unsigned long long>(
      std::numeric_limits<unsigned long>::max(), visited);
  test<Context, unsigned long long, ExpectedResultType, unsigned long long>(
      std::numeric_limits<unsigned long long>::max(), visited);

#ifndef TEST_HAS_NO_INT128
  test_handle<Context, __uint128_t, ExpectedResultType>(0, visited);
#endif // TEST_HAS_NO_INT128

  // Test floating point types.

  test<Context, float, ExpectedResultType, float>(-std::numeric_limits<float>::max(), visited);
  test<Context, float, ExpectedResultType, float>(-std::numeric_limits<float>::min(), visited);
  test<Context, float, ExpectedResultType, float>(-0.0, visited);
  test<Context, float, ExpectedResultType, float>(0.0, visited);
  test<Context, float, ExpectedResultType, float>(std::numeric_limits<float>::min(), visited);
  test<Context, float, ExpectedResultType, float>(std::numeric_limits<float>::max(), visited);

  test<Context, double, ExpectedResultType, double>(-std::numeric_limits<double>::max(), visited);
  test<Context, double, ExpectedResultType, double>(-std::numeric_limits<double>::min(), visited);
  test<Context, double, ExpectedResultType, double>(-0.0, visited);
  test<Context, double, ExpectedResultType, double>(0.0, visited);
  test<Context, double, ExpectedResultType, double>(std::numeric_limits<double>::min(), visited);
  test<Context, double, ExpectedResultType, double>(std::numeric_limits<double>::max(), visited);

  test<Context, long double, ExpectedResultType, long double>(-std::numeric_limits<long double>::max(), visited);
  test<Context, long double, ExpectedResultType, long double>(-std::numeric_limits<long double>::min(), visited);
  test<Context, long double, ExpectedResultType, long double>(-0.0, visited);
  test<Context, long double, ExpectedResultType, long double>(0.0, visited);
  test<Context, long double, ExpectedResultType, long double>(std::numeric_limits<long double>::min(), visited);
  test<Context, long double, ExpectedResultType, long double>(std::numeric_limits<long double>::max(), visited);

  // Test const CharT pointer types.

  test<Context, const CharT*, ExpectedResultType, const CharT*>(empty.c_str(), visited);
  test<Context, const CharT*, ExpectedResultType, const CharT*>(str.c_str(), visited);

  // Test string_view types.

  {
    using From = std::basic_string_view<CharT>;

    test_string_view<Context, ExpectedResultType>(From(), visited);
    test_string_view<Context, ExpectedResultType>(From(empty.c_str()), visited);
    test_string_view<Context, ExpectedResultType>(From(str.c_str()), visited);
  }
  {
    using From = std::basic_string_view<CharT, constexpr_char_traits<CharT>>;

    test_string_view<Context, ExpectedResultType>(From(), visited);
    test_string_view<Context, ExpectedResultType>(From(empty.c_str()), visited);
    test_string_view<Context, ExpectedResultType>(From(str.c_str()), visited);
  }

  // Test string types.

  {
    using From = std::basic_string<CharT>;

    test_string_view<Context, ExpectedResultType>(From(), visited);
    test_string_view<Context, ExpectedResultType>(From(empty.c_str()), visited);
    test_string_view<Context, ExpectedResultType>(From(str.c_str()), visited);
  }

  {
    using From = std::basic_string<CharT, constexpr_char_traits<CharT>, std::allocator<CharT>>;

    test_string_view<Context, ExpectedResultType>(From(), visited);
    test_string_view<Context, ExpectedResultType>(From(empty.c_str()), visited);
    test_string_view<Context, ExpectedResultType>(From(str.c_str()), visited);
  }

  {
    using From = std::basic_string<CharT, std::char_traits<CharT>, min_allocator<CharT>>;

    test_string_view<Context, ExpectedResultType>(From(), visited);
    test_string_view<Context, ExpectedResultType>(From(empty.c_str()), visited);
    test_string_view<Context, ExpectedResultType>(From(str.c_str()), visited);
  }

  {
    using From = std::basic_string<CharT, constexpr_char_traits<CharT>, min_allocator<CharT>>;

    test_string_view<Context, ExpectedResultType>(From(), visited);
    test_string_view<Context, ExpectedResultType>(From(empty.c_str()), visited);
    test_string_view<Context, ExpectedResultType>(From(str.c_str()), visited);
  }

  // Test pointer types.

  test<Context, const void*, ExpectedResultType>(nullptr, visited);
  int i = 0;
  test<Context, const void*, ExpectedResultType>(static_cast<void*>(&i), visited);
  const int ci = 0;
  test<Context, const void*, ExpectedResultType>(static_cast<const void*>(&ci), visited);
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
