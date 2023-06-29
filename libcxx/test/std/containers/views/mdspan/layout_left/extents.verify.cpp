//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class Extents>
// class layout_left::mapping;

// If Extents is not a specialization of extents, then the program is
// ill-formed.

// Mandates: If Extents::rank_dynamic() == 0 is true, then the size of the
// multidimensional index space Extents() is representable as a value of type
// typename Extents::index_type.

#include <mdspan>

void not_extents() {
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}layout_left::mapping template argument must be a specialization of extents}}
  [[maybe_unused]] std::layout_left::mapping<void> mapping;
}

void representable() {
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}layout_left::mapping product of static extents must be representable as index_type.}}
  [[maybe_unused]] std::layout_left::mapping<std::extents<char, 20, 20>> mapping;
}
