//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that __pointer_int_pair cannot be constant initialized with a value.
// This would mean that the constructor is `constexpr`, which should only be
// possible with compiler magic.

// UNSUPPORTED: c++03, c++11, c++14, c++17

// clang-format off

#include <__utility/pointer_int_pair.h>
#include <cstddef>

template <class Ptr, class UnderlyingType>
using single_bit_pair = std::__pointer_int_pair<Ptr, UnderlyingType, std::__integer_width{1}>;

constinit int ptr = 0;
constinit single_bit_pair<int*, size_t> continitiable_pointer_int_pair_values{&ptr, 0}; // expected-error {{variable does not have a constant initializer}}
