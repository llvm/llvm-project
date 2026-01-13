//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// This test verifies that the construction of locale::__imp does not allocate
// for facets, as it uses __sso_allocator<facet*, N>. It would fail if new
// facets have been added (using install()) but N hasn't been adjusted to
// account for them.

#include <cassert>

#include "count_new.h"

int main(int, char**) {
  assert(globalMemCounter.checkOutstandingNewEq(0));
  return 0;
}
