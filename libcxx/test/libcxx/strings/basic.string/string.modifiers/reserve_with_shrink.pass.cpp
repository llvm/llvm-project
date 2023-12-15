//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// modules builds should follow the new ABI
// UNSUPPORTED: clang-modules-build

#  include <__config>

// Check reserve(0) with old mangling shrinks for compatibility if it exists.
#if !defined(_LIBCPP_ABI_DO_NOT_RETAIN_SHRINKING_RESERVE)
#  define _LIBCPP_ENABLE_RESERVE_SHRINKING_ABI
#  include <string>
#  include <cassert>

#  include "test_macros.h"
#  include "min_allocator.h"

bool test() {
  std::string l = "Long string so that allocation definitely, for sure, absolutely happens. Probably.";
  const char* c = l.c_str();

  assert(l.__invariants());

  l.clear();
  assert(l.__invariants());
  assert(l.size() == 0);

  l.reserve(0);
  assert(l.__invariants());
  assert(l.size() == 0);
  assert(c != l.c_str());

  return true;
}
#else
bool test() { return true; }
#endif

int main(int, char**) {
  test();
  return 0;
}
