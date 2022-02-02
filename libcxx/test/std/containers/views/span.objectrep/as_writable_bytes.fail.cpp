//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// template <class ElementType, size_t Extent>
//     span<byte,
//          Extent == dynamic_extent
//              ? dynamic_extent
//              : sizeof(ElementType) * Extent>
//     as_writable_bytes(span<ElementType, Extent> s) noexcept;


#include <span>
#include <cassert>
#include <string>

#include "test_macros.h"

const int iArr2[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};

struct A {};

int main(int, char**)
{
    std::as_writable_bytes(std::span<const int>());            // expected-error {{no matching function for call to 'as_writable_bytes'}}
    std::as_writable_bytes(std::span<const long>());           // expected-error {{no matching function for call to 'as_writable_bytes'}}
    std::as_writable_bytes(std::span<const double>());         // expected-error {{no matching function for call to 'as_writable_bytes'}}
    std::as_writable_bytes(std::span<const A>());              // expected-error {{no matching function for call to 'as_writable_bytes'}}
    std::as_writable_bytes(std::span<const std::string>());    // expected-error {{no matching function for call to 'as_writable_bytes'}}

    std::as_writable_bytes(std::span<const int, 0>());         // expected-error {{no matching function for call to 'as_writable_bytes'}}
    std::as_writable_bytes(std::span<const long, 0>());        // expected-error {{no matching function for call to 'as_writable_bytes'}}
    std::as_writable_bytes(std::span<const double, 0>());      // expected-error {{no matching function for call to 'as_writable_bytes'}}
    std::as_writable_bytes(std::span<const A, 0>());           // expected-error {{no matching function for call to 'as_writable_bytes'}}
    std::as_writable_bytes(std::span<const std::string, (size_t)0>()); // expected-error {{no matching function for call to 'as_writable_bytes'}}

    std::as_writable_bytes(std::span<const int>   (iArr2, 1));     // expected-error {{no matching function for call to 'as_writable_bytes'}}
    std::as_writable_bytes(std::span<const int, 1>(iArr2 + 5, 1)); // expected-error {{no matching function for call to 'as_writable_bytes'}}

  return 0;
}
