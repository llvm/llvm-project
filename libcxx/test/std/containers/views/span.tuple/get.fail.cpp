//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <span>

// template <size_t _Ip, ElementType, size_t Extent>
//   constexpr ElementType& get(span<ElementType, Extent> s) noexcept;

#include <span>

#include "test_macros.h"


int main(int, char**)
{
//  No get for dynamic spans
    constexpr int arr [] = {0,1,2,3,4,5,6,7,8,9};
    (void) std::get< 0>(std::span<const int,  0>(arr, (size_t)0)); // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Index out of bounds in std::get<> (std::span)"}}
    (void) std::get< 5>(std::span<const int,  5>(arr,         5)); // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Index out of bounds in std::get<> (std::span)"}}
    (void) std::get<20>(std::span<const int, 10>(arr,        10)); // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Index out of bounds in std::get<> (std::span)"}}

  return 0;
}
