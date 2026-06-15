//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// template<class Extents>
// class layout_left_padded::mapping;

#include <cstdint>
#include <mdspan>

void not_extents() {
  // expected-error-re@*:* {{static assertion failed {{.*}}layout_left_padded::mapping template argument must be a specialization of extents}}
  [[maybe_unused]] std::layout_left_padded<4>::mapping<void> mapping;
}

void index_space_representable() {
  // expected-error@*:* {{layout_left_padded::mapping index space for static extents must be representable as index_type.}}
  [[maybe_unused]] std::layout_left_padded<4>::mapping<std::extents<int8_t, 20, 20>> mapping;
}

void padding_value_representable() {
  // expected-error@*:* {{layout_left_padded::mapping padding_value must be representable as index_type.}}
  [[maybe_unused]] std::layout_left_padded<128>::mapping<std::extents<int8_t, 1>> mapping;
}

void padding_stride_representable() {
  // expected-error@*:* {{layout_left_padded::mapping padded stride for the first static extent must be representable as size_t and index_type.}}
  [[maybe_unused]] std::layout_left_padded<4>::mapping<std::extents<int8_t, 127, 1>> mapping;
}

void padded_product_representable() {
  // expected-error@*:* {{layout_left_padded::mapping required span size for static extents must be representable as size_t and index_type.}}
  [[maybe_unused]] std::layout_left_padded<64>::mapping<std::extents<int8_t, 63, 2>> mapping;
}
