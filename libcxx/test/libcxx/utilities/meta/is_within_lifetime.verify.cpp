//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: gcc-15

// <type_traits>

// template<class U = void, class T>
//   consteval bool is_within_lifetime(const T* p) noexcept;
// Mandates: static_cast<const volatile U*>(p) is well-formed.

// LWG4138 <https://cplusplus.github.io/LWG/issue4138>
// std::is_within_lifetime shouldn't work when a function type is
// explicitly specified, even if it isn't evaluated

#include <type_traits>

template <class U, class T>
consteval bool checked_is_within_lifetime(T* p) {
  return p ? std::is_within_lifetime<U, T>(p) : false;
}
static_assert(!checked_is_within_lifetime<void, int>(nullptr));
static_assert(!checked_is_within_lifetime<void, void()>(nullptr));
// expected-error@*:* {{function pointer argument to '__builtin_is_within_lifetime' is not allowed}}

static_assert(!checked_is_within_lifetime<long, int>(nullptr));
// expected-error@*:* {{static_cast from 'const int *' to 'const volatile long *' is not allowed}}
static_assert(!checked_is_within_lifetime<int(), int>(nullptr));
// expected-error@*:* {{static_cast from 'const int *' to 'int (*)()' is not allowed}}

struct B {};
struct D1 : B {};
struct D2 : protected B {};
struct D3 : private B {};
struct D4 : D1, D2, D3 {};
struct D5 : virtual B {};

static_assert(!checked_is_within_lifetime<D2, B>(nullptr));
// expected-error@*:* {{cannot cast protected base class 'const B' to 'const volatile D2'}}
static_assert(!checked_is_within_lifetime<D3, B>(nullptr));
// expected-error@*:* {{cannot cast private base class 'const B' to 'const volatile D3'}}
static_assert(!checked_is_within_lifetime<D4, B>(nullptr));
// expected-error@*:* {{ambiguous cast from base 'B' to derived 'D4':}}
static_assert(!checked_is_within_lifetime<D5, B>(nullptr));
// expected-error@*:* {{cannot cast 'const B *' to 'const volatile D5 *' via virtual base 'B'}}
