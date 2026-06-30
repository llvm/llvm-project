//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++11

// <utility>

// template <class T1, class T2> struct pair

// template<class U1 = T1, class U2 = T2>
//   constexpr explicit(see below) pair(U1&& x, U2&& y);            // since C++11

// The constructor is defined as deleted if
// reference_constructs_from_temporary_v<first_type, U1&&> || reference_constructs_from_temporary_v<second_type, U2>
// is true. (since C++23)

// template<class U1, class U2>
//   constexpr explicit(see below) pair(pair<U1, U2>& p);           // since C++23
// template<class U1, class U2>
//   constexpr explicit(see below) pair(const pair<U1, U2>& p);     // since C++11
// template<class U1, class U2>
//   constexpr explicit(see below) pair(pair<U1, U2>&& p);          // since C++11
// template<class U1, class U2>
//   constexpr explicit(see below) pair(const pair<U1, U2>&& p);    // since C++23
// template<pair-like P>
//   constexpr explicit(see below) pair(P&& p);                     // since C++23

// The constructor is defined as deleted if
// reference_constructs_from_temporary_v<first_type, decltype(get<0>(FWD(p)))> ||
// reference_constructs_from_temporary_v<second_type, decltype(get<1>(FWD(p)))>
// is true. (since C++23)

// Such reference binding used to cause hard error for these constructors before C++23 due to CWG1696.

#include <array>
#include <complex>
#include <tuple>
#include <utility>

#include "test_macros.h"

void verify_two_arguments() {
  std::pair<const long&, int&&> p1{'a', 'b'};
#if TEST_STD_VER >= 23
  // expected-error@-2 {{call to deleted constructor of 'std::pair<const long &, int &&>'}}
#else
  // expected-error@*:* {{reference member 'first' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
  // expected-error@*:* {{reference member 'second' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
#endif

#if TEST_STD_VER >= 23
  std::pair<const long, int&&> p2({42L}, 'c');
  // expected-error@-1 {{call to deleted constructor of 'std::pair<const long, int &&>'}}
  std::pair<const long&, int> p3{'d', {}};
  // expected-error@-1 {{call to deleted constructor of 'std::pair<const long &, int>'}}
#endif
}

void verify_pair_const_lvalue() {
  const std::pair<char, int> src1{'a', 'b'};
  std::pair<const long&, const int&> dst1 = src1;
  (void)dst1;
#if TEST_STD_VER >= 23
  // expected-error@-3 {{invokes a deleted function}}
#else
  // expected-error@*:* {{reference member 'first' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
#endif

  const std::pair<long, char> src2{'a', 'b'};
  std::pair<const long&, const int&> dst2 = src2;
  (void)dst2;
#if TEST_STD_VER >= 23
  // expected-error@-3 {{conversion function from 'const pair<long, char>' to 'pair<const long &, const int &>' invokes a deleted function}}
#else
  // expected-error@*:* {{reference member 'second' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
#endif
}

void verify_pair_rvalue() {
  std::pair<char, int> src1{'a', 'b'};
  std::pair<const long&, int&&> dst1 = std::move(src1);
  (void)dst1;
#if TEST_STD_VER >= 23
  // expected-error@-3 {{invokes a deleted function}}
#else
  // expected-error@*:* {{reference member 'first' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
#endif

  std::pair<long, char> src2{'a', 'b'};
  std::pair<const long&, int&&> dst2 = std::move(src2);
  (void)dst2;
#if TEST_STD_VER >= 23
  // expected-error@-3 {{invokes a deleted function}}
#else
  // expected-error@*:* {{reference member 'second' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
#endif
}

#if TEST_STD_VER >= 23
void verify_pair_lvalue() {
  std::pair<char, int> src1{'a', 'b'};
  std::pair<const long&, int&> dst1 = src1; // expected-error {{invokes a deleted function}}

  std::pair<long, char> src2{'a', 'b'};
  std::pair<const long&, int&&> dst2 = src2; // expected-error {{invokes a deleted function}}
}

void verify_pair_const_rvalue() {
  const std::pair<char, int> src1{'a', 'b'};
  std::pair<const long&, const int&&> dst1 = std::move(src1); // expected-error {{invokes a deleted function}}

  const std::pair<long, char> src2{'a', 'b'};
  std::pair<const long&, const int&&> dst2 = std::move(src2); // expected-error {{invokes a deleted function}}
}

void verify_pair_like() {
  std::pair<const long&, int&&> p1  = std::make_tuple('a', int{'b'});  // expected-error {{invokes a deleted function}}
  std::pair<const long&, int&&> p2  = std::make_tuple(long{'a'}, 'b'); // expected-error {{invokes a deleted function}}
  std::pair<const char&, int&&> p3  = std::array<char, 2>{'a', 'b'};   // expected-error {{invokes a deleted function}}
  std::pair<const long&, char&&> p4 = std::array<char, 2>{'a', 'b'};   // expected-error {{invokes a deleted function}}

#  if TEST_STD_VER >= 26
  std::pair<const long double&, float&&> p5 = std::complex<float>{42.0f, 1729.0f};
  // expected-error@-1 {{invokes a deleted function}}
  std::pair<const float&, double&&> p6 = std::complex<float>{3.14159f, 2.71828f};
  // expected-error@-1 {{invokes a deleted function}}
#  endif
}
#endif

// Verify that copy-non-list-initialization ignores explicit but deleted overloads.
void verify_explicity() {
  struct ExplicitlyToInt {
    explicit operator int() const;
  };

  const std::pair<int, ExplicitlyToInt> src1;
  std::pair<int, int&&> dst1 = src1; // expected-error {{no viable conversion}}

  std::pair<int, ExplicitlyToInt> src2;
  std::pair<int, int&&> dst2 = std::move(src2); // expected-error {{no viable conversion}}

#if TEST_STD_VER >= 23
  const std::pair<int, ExplicitlyToInt> src3;
  std::pair<int, int&&> dst3 = std::move(src3); // expected-error {{no viable conversion}}

  std::pair<int, ExplicitlyToInt> src4;
  std::pair<int, int&&> dst4 = src4; // expected-error {{no viable conversion}}

  std::pair<int, int&&> dst5 = std::make_tuple(0, ExplicitlyToInt{}); // expected-error {{no viable conversion}}

  std::pair<int&&, int> dst6 = std::array<ExplicitlyToInt, 2>{}; // expected-error {{no viable conversion}}
#endif

#if TEST_STD_VER >= 26
  struct ExplicitlyFromFloat {
    explicit ExplicitlyFromFloat(float);
  };

  std::pair<ExplicitlyFromFloat, const ExplicitlyFromFloat&> dst7 = // expected-error {{no viable conversion}}
      std::complex<float>{};
#endif
}
