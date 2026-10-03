//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <format>

// [format.formatter.spec]:
// Each header that declares the template `formatter` provides the following
// enabled specializations:
// For each `charT`, for each cv-unqualified arithmetic type `ArithmeticT`
// other than char, wchar_t, char8_t, char16_t, or char32_t, a specialization
//    template<> struct formatter<ArithmeticT, charT>
//
// This file tests with `ArithmeticT = unsigned integer`, for each valid `charT`.
// Where `unsigned integer` is one of:
// - unsigned char
// - unsigned short
// - unsigned
// - unsigned long
// - unsigned long long
// - __uint128_t

#include <format>
#include <cassert>
#include <concepts>
#include <iterator>
#include <memory>
#include <type_traits>

#include "test_format_context.h"
#include "test_macros.h"
#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class StringT, class StringViewT, class ArithmeticT>
void test(StringT expected, StringViewT fmt, ArithmeticT arg, std::size_t offset) {
  using CharT = typename StringT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<ArithmeticT, CharT> formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  std::same_as<typename StringViewT::iterator> auto it = formatter.parse(parse_ctx);
  // std::to_address works around LWG3989 and MSVC STL's iterator debugging mechanism.
  assert(std::to_address(it) == std::to_address(fmt.end()) - offset);

  StringT result;
  auto out = std::back_inserter(result);
  using FormatCtxT = std::basic_format_context<decltype(out), CharT>;

  FormatCtxT format_ctx =
      test_format_context_create<decltype(out), CharT>(out, std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);
  assert(result == expected);
}

template <class StringT, class ArithmeticT>
void test_termination_condition(StringT expected, StringT f, ArithmeticT arg) {
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  using CharT = typename StringT::value_type;
  std::basic_string_view<CharT> fmt{f};
  assert(fmt.back() == CharT('}') && "Pre-condition failure");

  test(expected, fmt, arg, 1);
  fmt.remove_suffix(1);
  test(expected, fmt, arg, 0);
}

template <class ArithmeticT, class CharT>
void test_unsigned_integral_type() {
  using A = ArithmeticT;
  test_termination_condition(STR("0"), STR("}"), A(0));
  test_termination_condition(STR("255"), STR("}"), A(255));
  if (sizeof(A) > 1)
    test_termination_condition(STR("65535"), STR("}"), A(65535));
  if (sizeof(A) > 2)
    test_termination_condition(STR("4294967295"), STR("}"), A(4294967295));
  if (sizeof(A) > 4)
    test_termination_condition(STR("8446744073709551615"), STR("}"),
                               A(8446744073709551615));
#ifndef TEST_HAS_NO_INT128
  if (sizeof(A) > 8)
    test_termination_condition(
        STR("340282366920938463463374607431768211455"), STR("}"), A(std::numeric_limits<__uint128_t>::max()));
#endif
  // Test __formatter::__transform (libc++ specific).
  test_termination_condition(STR("FF"), STR("X}"), A(255));
}

template <class CharT>
void test_all_unsigned_integral_types() {
  test_unsigned_integral_type<unsigned char, CharT>();
  test_unsigned_integral_type<unsigned short, CharT>();
  test_unsigned_integral_type<unsigned, CharT>();
  test_unsigned_integral_type<unsigned long, CharT>();
  test_unsigned_integral_type<unsigned long long, CharT>();
#ifndef TEST_HAS_NO_INT128
  test_unsigned_integral_type<__uint128_t, CharT>();
#endif
}

#if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 256
// A _BitInt wider than 128 bits formats through the handle path and the charconv
// bignum conversion. 4096 bits is the headline width shared with the to_chars test.
template <class CharT>
void test_wide_unsigned_bitint() {
  {
    using A = unsigned _BitInt(256);
    test_termination_condition(STR("0"), STR("}"), A(0));
    test_termination_condition(STR("42"), STR("}"), A(42));
    test_termination_condition(
        STR("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
        STR("#x}"),
        std::numeric_limits<A>::max());
    test_termination_condition(
        STR("115792089237316195423570985008687907853269984665640564039457584007913129639935"),
        STR("}"),
        std::numeric_limits<A>::max());
  }
#  if __BITINT_MAXWIDTH__ >= 4096
  {
    using A = unsigned _BitInt(4096);
    test_termination_condition(
        STR("1044388881413152506691752710716624382579964249047383780384233483283953907971557456848826811934997558340890"
            "1067"
            "1443926283798757343818579360726323608785136527794595697654370999834036159013438371831442807001185594622637"
            "6318"
            "8393977127456723346843445866174968079087058037040712840487401186091144679777835980290066869389768817877859"
            "4690"
            "5630190260940599579453432823469303026696443059025015972399867714215541693835559885291486318237914434496734"
            "0878"
            "1187263949647510018904134900841706167509366833385055103297208826955076998361636941193301521379682583718809"
            "1833"
            "6567512213184928463681255502259983004123447848625956744921946170238065059132456108257318353800876086221028"
            "3427"
            "0197698202313169017678006675195485079921636419370285375124784014907159135459982790513399611551794271106831"
            "1340"
            "9058427288427979155484978295432353451706522326906139490598769300212296339568778287894844061600741294567491"
            "9823"
            "0505716423771548163213806310459029161369267083428564407304478999719017814657634732238502672530598997959960"
            "9079"
            "9469201774624817718449867455659250178329070473119433165550807568221846571746373296884912819520317457002440"
            "9266"
            "1691087414838507841192980452298185733897764810312608590300130241346718972667321649151113160292078173803343"
            "6090"
            "243804708340403154190335"),
        STR("}"),
        std::numeric_limits<A>::max());
  }
#  endif
}
#endif

int main(int, char**) {
  test_all_unsigned_integral_types<char>();
#if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 256
  test_wide_unsigned_bitint<char>();
#endif
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_all_unsigned_integral_types<wchar_t>();
#  if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 256
  test_wide_unsigned_bitint<wchar_t>();
#  endif
#endif

  return 0;
}
