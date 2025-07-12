//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Check that map::iterator is initialized when default constructed

#include <cassert>
#include <map>

template <class Iter>
void test() {
  Iter iter;
  Iter iter2 = Iter();
  assert(iter == iter2);
}

int main() {
  test<std::map<int, int>::iterator>();
  test<std::map<int, int>::const_iterator>();
}
