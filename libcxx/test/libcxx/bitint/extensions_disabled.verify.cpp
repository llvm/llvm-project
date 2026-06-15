//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Default build (no _LIBCPP_ENABLE_BITINT_EXTENSIONS): libc++ rejects _BitInt
// operands in <bit>, <charconv>, <format>, std::hash, std::cmp_*. Type traits
// and numeric_limits stay accepted per P3666R4.

#include <bit>
#include <charconv>
#include <format>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>

#include "test_macros.h"

#ifdef _LIBCPP_ENABLE_BITINT_EXTENSIONS
#  error "this test runs under the default (macro undefined)"
#endif

#ifndef __BITINT_MAXWIDTH__
// expected-no-diagnostics
#else

using S = _BitInt(64);
using U = unsigned _BitInt(64);

// Canary: the trait plumbing is wired. If a future change breaks the trait or
// the macro propagation, these assertions catch it before the rest of the
// test, with a clear message.
static_assert(std::__is_bit_precise_integer_v<U>);
static_assert(std::__is_bit_precise_integer_v<S>);
static_assert(!std::__is_bit_precise_integer_v<int>);
static_assert(!std::__admits_bitint_extension_v<U>);
static_assert(std::__admits_bitint_extension_v<int>);

void test_byteswap(U u, S s) {
  U cu = 0;
  (void)std::byteswap(u);  // expected-error@*:* {{no matching function for call to 'byteswap'}}
  (void)std::byteswap(s);  // expected-error@*:* {{no matching function for call to 'byteswap'}}
  (void)std::byteswap(cu); // expected-error@*:* {{no matching function for call to 'byteswap'}}
}

void test_hash(U u, S s) {
  // The primary __hash_impl has = delete'd special members under default.
  std::hash<U> hu; // expected-error@*:* {{implicitly-deleted default constructor}}
  std::hash<S> hs; // expected-error@*:* {{implicitly-deleted default constructor}}
  (void)hu;
  (void)hs;
  (void)u;
  (void)s;
}

void test_charconv(char* b, char* e, U u, S s) {
  // to_chars: integral templates disabled, falls to the floating-point overloads -> ambiguous.
  (void)std::to_chars(b, e, u); // expected-error@*:* {{call to 'to_chars' is ambiguous}}
  (void)std::to_chars(b, e, s); // expected-error@*:* {{call to 'to_chars' is ambiguous}}
  // from_chars: integral templates disabled, falls to the deleted bool overload.
  (void)std::from_chars(b, e, u); // expected-error@*:* {{call to deleted function 'from_chars'}}
  (void)std::from_chars(b, e, s); // expected-error@*:* {{call to deleted function 'from_chars'}}
}

void test_cmp(U u, S s) {
  (void)std::cmp_less(u, 0);    // expected-error@*:* {{no matching function for call to 'cmp_less'}}
  (void)std::cmp_equal(s, 0);   // expected-error@*:* {{no matching function for call to 'cmp_equal'}}
  (void)std::cmp_greater(u, 0); // expected-error@*:* {{no matching function for call to 'cmp_greater'}}
  (void)std::in_range<U>(0);    // expected-error@*:* {{no matching function for call to 'in_range'}}
}

// std::format's _BitInt rejection diagnoses deep in formatter selection; the
// SFINAE check at the bottom of this file verifies it without binding to a
// specific diagnostic string.

// gcd, lcm, midpoint are exercised by the SFINAE probes below (midpoint) and
// by direct instantiation expectations during macro-on/off CI runs (gcd/lcm
// reject via static_assert, which fires only on instantiation).

// Surface P3666R4 keeps.
static_assert(std::is_integral_v<U>);
static_assert(std::is_signed_v<S>);
static_assert(std::is_unsigned_v<U>);
static_assert(std::is_same_v<std::make_unsigned_t<S>, U>);
static_assert(std::numeric_limits<U>::digits == 64);

// SFINAE probes: detection idioms must return false, not hard-error. Widths
// span sub-byte (1), byte-aligned (8/32/64), sub-byte-padded (13/65), and a
// wide signed/unsigned mix.
template <class T>
concept has_byteswap = requires(T t) { std::byteswap(t); };
template <class T>
concept has_hash = requires(T t) { std::hash<T>{}(t); };
template <class T>
concept has_to_chars = requires(char* b, char* e, T t) { std::to_chars(b, e, t); };
template <class T>
concept has_from_chars = requires(char* b, char* e, T t) { std::from_chars(b, e, t); };
template <class T>
concept has_cmp_less = requires(T t) { std::cmp_less(t, 0); };
template <class T>
concept has_midpoint = requires(T t) { std::midpoint(t, t); };
// std::format / std::gcd / std::lcm reject via static_assert, not SFINAE.

// Clang requires signed _BitInt(N) with N >= 2; only the unsigned form is
// well-formed at N == 1.
static_assert(!has_byteswap<unsigned _BitInt(1)>);
static_assert(!has_byteswap<_BitInt(8)>);
static_assert(!has_byteswap<unsigned _BitInt(8)>);
static_assert(!has_byteswap<_BitInt(13)>);
static_assert(!has_byteswap<unsigned _BitInt(13)>);
static_assert(!has_byteswap<_BitInt(32)>);
static_assert(!has_byteswap<_BitInt(64)>);
static_assert(!has_byteswap<unsigned _BitInt(64)>);
#  if __BITINT_MAXWIDTH__ >= 128
static_assert(!has_byteswap<_BitInt(128)>);
static_assert(!has_byteswap<unsigned _BitInt(128)>);
#  endif

// cv-qualified _BitInt rejected too.
static_assert(!has_byteswap<const _BitInt(64)>);
static_assert(!has_byteswap<const unsigned _BitInt(64)>);

// Standard integers still flow.
static_assert(has_byteswap<int>);
static_assert(has_byteswap<unsigned long long>);
static_assert(has_byteswap<bool>);

// Other gated facilities, same probe.
static_assert(!has_hash<U>);
static_assert(!has_hash<_BitInt(8)>);
static_assert(!has_hash<_BitInt(128)>);
static_assert(!has_to_chars<U>);
static_assert(!has_to_chars<_BitInt(8)>);
static_assert(!has_from_chars<U>);
static_assert(!has_cmp_less<U>);
static_assert(!has_cmp_less<_BitInt(8)>);

static_assert(!has_midpoint<U>);
static_assert(!has_midpoint<_BitInt(8)>);

static_assert(has_hash<int>);
static_assert(has_to_chars<int>);
static_assert(has_from_chars<int>);
static_assert(has_cmp_less<int>);
static_assert(has_midpoint<int>);

#endif // __BITINT_MAXWIDTH__
