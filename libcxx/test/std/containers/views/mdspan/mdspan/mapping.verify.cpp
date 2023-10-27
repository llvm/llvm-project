//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class ElementType, class Extents, class LayoutPolicy = layout_right, class AccessorPolicy = default_accessor>
// class mdspan;
//
// Mandates:
//  - LayoutPolicy shall meet the layout mapping policy requirements ([mdspan.layout.policy.reqmts])

#include <mdspan>

void not_layout_policy() {
  // expected-error-re@*:* {{static assertion failed {{.*}}mdspan: LayoutPolicy template parameter is invalid. A common mistake is to pass a layout mapping instead of a layout policy}}
  [[maybe_unused]] std::mdspan<int, std::extents<int>, std::layout_left::template mapping<std::extents<int>>> m;
}
