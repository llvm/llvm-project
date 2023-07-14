// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#include <mdspan>

export module std:mdspan;
export namespace std {
  // [mdspan.extents], class template extents
  using std::extents;

  // [mdspan.extents.dextents], alias template dextents
  using std::dextents;

  // [mdspan.layout], layout mapping
  using std::layout_left;
  using std::layout_right;
  // using std::layout_stride;

  // [mdspan.accessor.default], class template default_accessor
  using std::default_accessor;

  // [mdspan.mdspan], class template mdspan
  // using std::mdspan;
} // namespace std
