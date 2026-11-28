//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HIVE_HELPERS_H
#define SUPPORT_HIVE_HELPERS_H

#include <cassert>
#include <hive>

#include "test_macros.h"

template <class T, class Allocator>
void check_invariant(const std::hive<T, Allocator>& h) {
  LIBCPP_ASSERT(h.__invariants());
  assert(h.empty() == (h.size() == 0));
  assert(h.size() <= h.capacity());
  assert((std::hive<T, Allocator>::is_within_hard_limits(h.block_capacity_limits())));
}

template <class T, class Allocator>
void check_empty(const std::hive<T, Allocator>& h, std::hive_limits expected) {
  check_invariant(h);
  assert(h.empty());
  assert(h.size() == 0);
  assert(h.capacity() == 0);

  const std::hive_limits l = h.block_capacity_limits();
  assert(l.min == expected.min);
  assert(l.max == expected.max);
}

#endif // SUPPORT_HIVE_HELPERS_H
