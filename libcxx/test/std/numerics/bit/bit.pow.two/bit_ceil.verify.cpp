//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T>
//   constexpr T bit_ceil(T x) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//          T is an unsigned integer type

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

class A{};
enum       E1 : unsigned char { rEd };
enum class E2 : unsigned char { red };

template <typename T>
constexpr bool toobig()
{
    return 0 == std::bit_ceil(std::numeric_limits<T>::max());
}

int main(int, char**)
{
    // Make sure we generate a compile-time error for UB
    static_assert(toobig<unsigned char>(),      ""); // expected-error {{static assertion expression is not an integral constant expression}}
    static_assert(toobig<unsigned short>(),     ""); // expected-error {{static assertion expression is not an integral constant expression}}
    static_assert(toobig<unsigned>(),           ""); // expected-error {{static assertion expression is not an integral constant expression}}
    static_assert(toobig<unsigned long>(),      ""); // expected-error {{static assertion expression is not an integral constant expression}}
    static_assert(toobig<unsigned long long>(), ""); // expected-error {{static assertion expression is not an integral constant expression}}

    static_assert(toobig<std::uint8_t>(), "");   // expected-error {{static assertion expression is not an integral constant expression}}
    static_assert(toobig<std::uint16_t>(), "");  // expected-error {{static assertion expression is not an integral constant expression}}
    static_assert(toobig<std::uint32_t>(), "");  // expected-error {{static assertion expression is not an integral constant expression}}
    static_assert(toobig<std::uint64_t>(), "");  // expected-error {{static assertion expression is not an integral constant expression}}
    static_assert(toobig<std::size_t>(), "");    // expected-error {{static assertion expression is not an integral constant expression}}
    static_assert(toobig<std::uintmax_t>(), ""); // expected-error {{static assertion expression is not an integral constant expression}}
    static_assert(toobig<std::uintptr_t>(), ""); // expected-error {{static assertion expression is not an integral constant expression}}

    return 0;
}
