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
// This file tests with `ArithmeticT = signed integer`, for each valid `charT`.
// Where `signed integer` is one of:
// - signed char
// - short
// - int
// - long
// - long long
// - __int128_t

#include <format>
#include <cassert>
#include <concepts>
#include <cstdint>
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

template <class Arithmetic, class CharT>
void test_signed_integral_type() {
  using A = Arithmetic;
  test_termination_condition(STR("-128"), STR("}"), A(-128));
  test_termination_condition(STR("0"), STR("}"), A(0));
  test_termination_condition(STR("127"), STR("}"), A(127));
  if (sizeof(A) > 1) {
    test_termination_condition(STR("-32768"), STR("}"), A(-32768));
    test_termination_condition(STR("32767"), STR("}"), A(32767));
  }
  if (sizeof(A) > 2) {
    test_termination_condition(STR("-2147483648"), STR("}"), A(-2147483648));
    test_termination_condition(STR("2147483647"), STR("}"), A(2147483647));
  }
  if (sizeof(A) > 4) {
    test_termination_condition(STR("-9223372036854775808"), STR("}"), A(std::numeric_limits<std::int64_t>::min()));
    test_termination_condition(STR("9223372036854775807"), STR("}"), A(std::numeric_limits<std::int64_t>::max()));
  }
#ifndef TEST_HAS_NO_INT128
  if (sizeof(A) > 8) {
    test_termination_condition(
        STR("-170141183460469231731687303715884105728"), STR("}"), A(std::numeric_limits<__int128_t>::min()));
    test_termination_condition(
        STR("170141183460469231731687303715884105727"), STR("}"), A(std::numeric_limits<__int128_t>::max()));
  }
#endif
}

template <class CharT>
void test_all_signed_integral_types() {
  test_signed_integral_type<signed char, CharT>();
  test_signed_integral_type<short, CharT>();
  test_signed_integral_type<int, CharT>();
  test_signed_integral_type<long, CharT>();
  test_signed_integral_type<long long, CharT>();
#ifndef TEST_HAS_NO_INT128
  test_signed_integral_type<__int128_t, CharT>();
#endif
}

#if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 256
// A _BitInt wider than 128 bits formats through the handle path and the charconv
// bignum conversion. 4096 bits is the headline width shared with the to_chars test.
template <class CharT>
void test_wide_signed_bitint() {
  {
    using A = signed _BitInt(256);
    test_termination_condition(STR("0"), STR("}"), A(0));
    test_termination_condition(STR("-42"), STR("}"), A(-42));
    test_termination_condition(STR("FF"), STR("X}"), A(255));
    test_termination_condition(
        STR("-57896044618658097711785492504343953926634992332820282019728792003956564819968"),
        STR("}"),
        std::numeric_limits<A>::min());
    test_termination_condition(
        STR("57896044618658097711785492504343953926634992332820282019728792003956564819967"),
        STR("}"),
        std::numeric_limits<A>::max());
  }
#  if __BITINT_MAXWIDTH__ >= 4096
  {
    using A = signed _BitInt(4096);
    test_termination_condition(
        STR("-522194440706576253345876355358312191289982124523691890192116741641976953985778728424413405967498779170445"
            "0533"
            "5721963141899378671909289680363161804392568263897297848827185499917018079506719185915721403500592797311318"
            "8159"
            "4196988563728361673421722933087484039543529018520356420243700593045572339888917990145033434694884408938929"
            "7345"
            "2815095130470299789726716411734651513348221529512507986199933857107770846917779942645743159118957217248367"
            "0439"
            "0593631974823755009452067450420853083754683416692527551648604413477538499180818470596650760689841291859404"
            "5916"
            "8283756106592464231840627751129991502061723924312978372460973085119032529566228054128659176900438043110514"
            "1713"
            "5098849101156584508839003337597742539960818209685142687562392007453579567729991395256699805775897135553415"
            "5670"
            "4529213644213989577742489147716176725853261163453069745299384650106148169784389143947422030800370647283745"
            "9911"
            "5252858211885774081606903155229514580684633541714282203652239499859508907328817366119251336265299498979980"
            "4539"
            "9734600887312408859224933727829625089164535236559716582775403784110923285873186648442456409760158728501220"
            "4633"
            "0845543707419253920596490226149092866948882405156304295150065120673359486333660824575556580146039086901671"
            "8045"
            "121902354170201577095168"),
        STR("}"),
        std::numeric_limits<A>::min());
    test_termination_condition(
        STR("5221944407065762533458763553583121912899821245236918901921167416419769539857787284244134059674987791704450"
            "5335"
            "7219631418993786719092896803631618043925682638972978488271854999170180795067191859157214035005927973113188"
            "1594"
            "1969885637283616734217229330874840395435290185203564202437005930455723398889179901450334346948844089389297"
            "3452"
            "8150951304702997897267164117346515133482215295125079861999338571077708469177799426457431591189572172483670"
            "4390"
            "5936319748237550094520674504208530837546834166925275516486044134775384991808184705966507606898412918594045"
            "9168"
            "2837561065924642318406277511299915020617239243129783724609730851190325295662280541286591769004380431105141"
            "7135"
            "0988491011565845088390033375977425399608182096851426875623920074535795677299913952566998057758971355534155"
            "6704"
            "5292136442139895777424891477161767258532611634530697452993846501061481697843891439474220308003706472837459"
            "9115"
            "2528582118857740816069031552295145806846335417142822036522394998595089073288173661192513362652994989799804"
            "5399"
            "7346008873124088592249337278296250891645352365597165827754037841109232858731866484424564097601587285012204"
            "6330"
            "8455437074192539205964902261490928669488824051563042951500651206733594863336608245755565801460390869016718"
            "0451"
            "21902354170201577095167"),
        STR("}"),
        std::numeric_limits<A>::max());
  }
#  endif
}
#endif

int main(int, char**) {
  test_all_signed_integral_types<char>();
#if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 256
  test_wide_signed_bitint<char>();
#endif
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_all_signed_integral_types<wchar_t>();
#  if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 256
  test_wide_signed_bitint<wchar_t>();
#  endif
#endif

  return 0;
}
