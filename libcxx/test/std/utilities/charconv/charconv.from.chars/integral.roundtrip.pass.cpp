//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=12712420

// <charconv>

// constexpr from_chars_result from_chars(const char* first, const char* last,
//                                        Integral& value, int base = 10)

#include <charconv>
#include <system_error>

#include "test_macros.h"
#include "charconv_test_helpers.h"

template <typename T>
struct test_basics : roundtrip_test_base<T>
{
    using roundtrip_test_base<T>::test;

    TEST_CONSTEXPR_CXX23 void operator()()
    {
        test(0);
        test(42);
        test(32768);
        test(0, 10);
        test(42, 10);
        test(32768, 10);
        test(0xf, 16);
        test(0xdeadbeaf, 16);
        test(0755, 8);

        for (int b = 2; b < 37; ++b)
        {
            using xl = std::numeric_limits<T>;

            test(1, b);
            test(-1, b);
            test(xl::lowest(), b);
            test((xl::max)(), b);
            test((xl::max)() / 2, b);
        }
    }
};

template <typename T>
struct test_signed : roundtrip_test_base<T>
{
    using roundtrip_test_base<T>::test;

    TEST_CONSTEXPR_CXX23 void operator()()
    {
        test(-1);
        test(-12);
        test(-1, 10);
        test(-12, 10);
        test(-21734634, 10);
        test(-2647, 2);
        test(-0xcc1, 16);

        for (int b = 2; b < 37; ++b)
        {
            using xl = std::numeric_limits<T>;

            test(0, b);
            test(xl::lowest(), b);
            test((xl::max)(), b);
        }
    }
};

#if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 4096
// Round-trip is the broad-coverage backbone for wide _BitInt: to_chars then
// from_chars must recover the value and consume exactly what was written.
template <int Bits>
TEST_CONSTEXPR_CXX23 void rt_one(unsigned _BitInt(Bits) u, int base) {
  char buf[Bits + 2]; // base 2 is the widest: up to Bits digits, plus a sign

  std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), u, base);
  assert(r.ec == std::errc{});
  unsigned _BitInt(Bits) ub = 1;
  std::from_chars_result f  = std::from_chars(buf, r.ptr, ub, base);
  assert(f.ec == std::errc{} && f.ptr == r.ptr && ub == u);

  signed _BitInt(Bits) s = static_cast<signed _BitInt(Bits)>(u);
  r                      = std::to_chars(buf, buf + sizeof(buf), s, base);
  assert(r.ec == std::errc{});
  signed _BitInt(Bits) sb = 1;
  f                       = std::from_chars(buf, r.ptr, sb, base);
  assert(f.ec == std::errc{} && f.ptr == r.ptr && sb == s);
}

template <int Bits>
TEST_CONSTEXPR_CXX23 void rt_width() {
  using U = unsigned _BitInt(Bits);
  using S = signed _BitInt(Bits);
  for (int b = 2; b <= 36; ++b) {
    rt_one<Bits>(U(0), b);
    rt_one<Bits>(U(1), b);
    rt_one<Bits>(std::numeric_limits<U>::max(), b);                 // all ones
    rt_one<Bits>(static_cast<U>(std::numeric_limits<S>::max()), b); // signed max
    rt_one<Bits>(static_cast<U>(std::numeric_limits<S>::min()), b); // signed min pattern
    rt_one<Bits>(std::numeric_limits<U>::max() / 3, b);             // arbitrary spread
  }
}

// Byte-aligned widths only, kept small enough for constant evaluation.
TEST_CONSTEXPR_CXX23 bool test_wide() {
  rt_width<256>();
  return true;
}

// 4096 bits is the headline "reasonable width": exercise it in constant
// evaluation too, on a representative base/value subset. The full base sweep at
// this width is kept at runtime (test_wide_runtime) to bound constexpr cost.
TEST_CONSTEXPR_CXX23 bool test_wide_4096() {
  using U        = unsigned _BitInt(4096);
  using S        = signed _BitInt(4096);
  const U vals[] = {
      U(0),
      U(1),
      std::numeric_limits<U>::max(),                 // all ones
      static_cast<U>(std::numeric_limits<S>::min()), // signed min pattern
      std::numeric_limits<U>::max() / 3,             // arbitrary spread
  };
  for (U v : vals) {
    rt_one<4096>(v, 2);
    rt_one<4096>(v, 10);
    rt_one<4096>(v, 16);
  }
  return true;
}

// Independent oracle for a huge 4096-bit value. The decimal and hex strings are
// computed by Python's arbitrary-precision int, so they catch a bug that
// to_chars and from_chars might share, which a pure round-trip cannot. The value
// is built from its bits, independent of the strings; the sparse pattern gives
// long interior-zero digit runs plus a few clusters.
TEST_CONSTEXPR_CXX23 bool test_wide_oracle() {
  using U   = unsigned _BitInt(4096);
  const U v = (U(1) << 4095) | (U(1) << 3000) | (U(1) << 1500) | (U(1) << 700) | U(0xDEADBEEFCAFEULL);

  const char dec[] =
      "522194440706576253345876355358312191289982124523691890192116741641976953985778728424413405967498779170445053357"
      "219631418993786719092896803631618043925682638972978488271854999170180795067191859157214035005927973113188159419"
      "698856372836167342172293308748403954352901852035642024370059304557233988891799014503343469488440893892973454045"
      "327052631416966658275225011404027988935425249341701966051886267829647532452891421103226957550676830796693159988"
      "124656750434275418911327793613319612182498160234866841889152188408311919545372889132335138169799010480395144069"
      "285672659513941271465489792446864983055373122330734549035017206965872628152160774541392717285230405353948204585"
      "721558937253615907178075626191061140618987126175926194937994062055639754879354953746504582267597824430780046407"
      "654767442218179275002597906567544454462957002216329001106694021676565785487186595814555929076464240677469561177"
      "107218357758641620381943183829806874856274226199817314163151233513560130957415780849310362681536620722609148912"
      "812834526196870692951390912269511897170617673817942753906754404005592596448147882135379795095255338256287637038"
      "130680736166406732152594023784824442544596004877042110465174074339660336326179460333127852406170867605207327911"
      "022565509886";
  const char hex[] =
      "80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000"
      "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
      "0000000000000000000000deadbeefcafe";
  // Base 36 exercises the generic loop's full 0-9a-z alphabet, which the base-10
  // and base-16 (LUT) checks above do not reach.
  const char b36[] =
      "1c58ccvc205beuqph4wbfhcolylc3mogv6cfu44q0ny092hqw421ukr2pnoci21sapp8q00xd2kotzuf65r9stubez1tfplsprvrzegyah5l9h"
      "71zr6zddlj4112jbkymn7hv9porwxfhp8jmu3gjgabuvjoc0t3p140isx85z3k4yuaoxq4r9wmummh8rujjxrazfodidhk9o6apnne4pacu79i"
      "n1cf5qaedb7s1dpskq2r9wgdswck57yqvobgs1d2xg154sb95jg39tt0bzyf5aptbw9z7ahqjnt3w7jp1tth21xwfn8dgzb5wvlc3zeijz0asb"
      "5d38l7qxbsj8pg8tkibhs3ooie5aizfh2657d6tz8ysut8g3zm254dqifpqj1i91wmfkofgvks973a706bz0ejpjdf5ucmra7d1x9x9wuiz044"
      "g7jcu71ofwdn4iwk8sxj17n3wrucuxed308dhf5aqumg19q7hftzrmp3obtygzt0zmut76adjffjwsmh3q5wcmx11d0rvigxuklx1i4r6stzuq"
      "y2fzxxhwberb6qmssvqffyt0mtzpaiez6s2p0api2i57fb6sfjkdj0c9ofdlnwo852fauo2bcxgtz62qe2ym2gt4g25haatt7xm9ck7wf2wwg7"
      "j9gpc8774a0u1p3wrolmysbzqagimzsg5foifbriuwyqkjlysp6qe3tvy658zmwpk2ib8p7b063v079ue0un8k61d7cabadmbjhki1c7g02y7d"
      "t6wrr3szp6wzcmcfe8tw16m";

  char buf[4100];
  std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), v, 10);
  assert(r.ec == std::errc{} && static_cast<std::size_t>(r.ptr - buf) == sizeof(dec) - 1);
  for (std::size_t i = 0; i + 1 < sizeof(dec); ++i)
    assert(buf[i] == dec[i]);

  r = std::to_chars(buf, buf + sizeof(buf), v, 16);
  assert(r.ec == std::errc{} && static_cast<std::size_t>(r.ptr - buf) == sizeof(hex) - 1);
  for (std::size_t i = 0; i + 1 < sizeof(hex); ++i)
    assert(buf[i] == hex[i]);

  r = std::to_chars(buf, buf + sizeof(buf), v, 36);
  assert(r.ec == std::errc{} && static_cast<std::size_t>(r.ptr - buf) == sizeof(b36) - 1);
  for (std::size_t i = 0; i + 1 < sizeof(b36); ++i)
    assert(buf[i] == b36[i]);

  U back                   = 0;
  std::from_chars_result f = std::from_chars(dec, dec + sizeof(dec) - 1, back, 10);
  assert(f.ec == std::errc{} && f.ptr == dec + sizeof(dec) - 1 && back == v);

  back = 0;
  f    = std::from_chars(hex, hex + sizeof(hex) - 1, back, 16);
  assert(f.ec == std::errc{} && f.ptr == hex + sizeof(hex) - 1 && back == v);

  back = 0;
  f    = std::from_chars(b36, b36 + sizeof(b36) - 1, back, 36);
  assert(f.ec == std::errc{} && f.ptr == b36 + sizeof(b36) - 1 && back == v);
  return true;
}

// Wider and non-byte-aligned widths at runtime. Non-aligned widths are excluded
// from constant evaluation pending the clang-23 fix for constexpr
// __builtin_mul_overflow on non-byte-aligned _BitInt (llvm.org/PR204085).
void test_wide_runtime() {
  rt_width<512>();
  rt_width<129>();
  rt_width<257>();
#  if __SIZEOF_POINTER__ >= 8
  // The widest sweeps are quadratic and, with 32-bit limbs, overrun the lit
  // timeout on a 32-bit target (seen on clang-21 i686-mingw). 64-bit configs
  // cover them; 32-bit keeps the smaller wide widths above.
  rt_width<1000>();
  rt_width<4096>();
#  endif
}
#endif

TEST_CONSTEXPR_CXX23 bool test()
{
    run<test_basics>(integrals);
    run<test_signed>(all_signed);

    return true;
}

int main(int, char**) {
    test();
#if TEST_STD_VER > 20
    static_assert(test());
#endif

#if defined(__BITINT_MAXWIDTH__) && __BITINT_MAXWIDTH__ >= 4096
    test_wide();
    test_wide_runtime();
    // 4096-bit round-trips (runtime and constant-evaluated) run on 64-bit targets
    // only. On a 32-bit target the quadratic base-10 conversion is slow enough to
    // exceed the lit timeout (seen on clang-21 i686-mingw); the smaller wide
    // widths in test_wide / test_wide_runtime still cover 32-bit.
#  if __SIZEOF_POINTER__ >= 8
    test_wide_4096();
    test_wide_oracle();
#  endif
#  if TEST_STD_VER > 20
    static_assert(test_wide());
#    if __SIZEOF_POINTER__ >= 8
    static_assert(test_wide_4096());
    static_assert(test_wide_oracle());
#    endif
#  endif
#endif

    return 0;
}
