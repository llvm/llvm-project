// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fconstexpr-steps=100 -fsyntax-only -verify %s
// expected-no-diagnostics

// Pin x86_64 so the wide _BitInt does not depend on the host target's
// __BITINT_MAXWIDTH__ (e.g. AArch64 caps it at 128).

// The builtin does not spend constexpr steps proportional to the operand width.
// Parsing the 302-digit decimal of 2^1000 into a 1024-bit _BitInt under a
// 100-step budget succeeds; a hand-written Horner loop over 302 digits would
// exceed it. to_chars cannot be shown the same way, because its output buffer
// is a constexpr array whose bound is itself limited by -fconstexpr-steps.

using u1024 = unsigned _BitInt(1024);

constexpr u1024 parse(const char *s, int n) {
  u1024 out = 0;
  int ec = 0;
  __builtin_from_chars(s, s + n, &out, 10, &ec);
  return ec == 0 ? out : 0;
}

static_assert(
    parse("107150860718626732094842504906000181056140481170553360744375038837"
          "035105112493612249319837881569585812759467291755314682518714528569"
          "231404359845775746985748039345677748242309854210746050623711418779"
          "541821530464749835819412673987675591655439460770629145711964776865"
          "42167660429831652624386837205668069376",
          302) == ((u1024)1) << 1000);
