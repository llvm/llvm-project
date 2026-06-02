//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// template<class ElementType, size_t Extent>
//   span<const byte, Extent == dynamic_extent ? dynamic_extent : sizeof(ElementType) * Extent>
//     as_bytes(span<ElementType, Extent> s) noexcept;
//
// Constraints:
//   is_volatile_v<ElementType> is false.

#include <span>
#include <string>

#include "test_macros.h"

struct A {};

void f() {
  std::as_bytes(std::span<volatile int>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<volatile long>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<volatile double>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<volatile A>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<volatile std::string>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}

  std::as_bytes(std::span<const volatile int>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<const volatile long>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<const volatile double>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<const volatile A>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<const volatile std::string>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}

  std::as_bytes(std::span<volatile int, 0>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<volatile long, 0>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<volatile double, 0>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<volatile A, 0>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<volatile std::string, (std::size_t)0>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}

  std::as_bytes(std::span<const volatile int, 0>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<const volatile long, 0>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<const volatile double, 0>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<const volatile A, 0>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
  std::as_bytes(std::span<const volatile std::string, (std::size_t)0>());
  // expected-error@-1 {{no matching function for call to 'as_bytes'}}
}
