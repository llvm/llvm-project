//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<size_t N, class T>
// [[nodiscard]] constexpr T* assume_aligned(T* ptr);

#include <memory>
#include <cstddef>

template <size_t Size>
constexpr bool test() {
  char data[1];

  [[maybe_unused]] auto data1 = std::assume_aligned<Size>(data);

  return true;
}

static_assert(test<2>());
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
// expected-note@* {{alignment of the base pointee object (1 byte) is less than the asserted 2 bytes}}

static_assert(test<4>());
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
// expected-note@* {{alignment of the base pointee object (1 byte) is less than the asserted 4 bytes}}

static_assert(test<8>());
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
// expected-note@* {{alignment of the base pointee object (1 byte) is less than the asserted 8 bytes}}

static_assert(test<16>());
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
// expected-note@* {{alignment of the base pointee object (1 byte) is less than the asserted 16 bytes}}

static_assert(test<32>());
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
// expected-note@* {{alignment of the base pointee object (1 byte) is less than the asserted 32 bytes}}

static_assert(test<64>());
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
// expected-note@* {{alignment of the base pointee object (1 byte) is less than the asserted 64 bytes}}
