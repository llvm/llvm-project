//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <mdspan>

// template<class ElementType, size_t ByteAlignment>
// class aligned_accessor;

// ByteAlignement is required to be a power of two and greater or equal to alignof(ElementType).

#include <mdspan>

void not_power_of_two() {
  // expected-error-re@*:* {{static assertion failed {{.*}}aligned_accessor: byte alignment must be a power of two}}
  [[maybe_unused]] std::aligned_accessor<int, 12> acc;
}

struct alignas(8) S {};

void insufficiently_aligned() {
  // expected-error-re@*:* {{static assertion failed {{.*}}aligned_accessor: insufficient byte alignment}}
  [[maybe_unused]] std::aligned_accessor<S, 4> acc;
}
