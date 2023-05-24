// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#if __has_include(<flat_set>)
#  error "include this header unconditionally and uncomment the exported symbols"
#  include <flat_set>
#endif

export module std:flat_set;
export namespace std {
#if 0
  // [flat.set], class template flat_­set
  using std::flat_set;

  using std::sorted_unique;
  using std::sorted_unique_t;

  using std::uses_allocator;

  // [flat.set.erasure], erasure for flat_­set
  using std::erase_if;

  // [flat.multiset], class template flat_­multiset
  using std::flat_multiset;

  using std::sorted_equivalent;
  using std::sorted_equivalent_t;
#endif
} // namespace std
