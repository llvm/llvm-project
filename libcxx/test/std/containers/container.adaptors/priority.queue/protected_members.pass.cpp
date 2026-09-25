//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// template <class T, class Container = vector<T>,
//           class Compare = less<typename Container::value_type>>
// class priority_queue
// {
// protected:
//     container_type c;
//     Compare comp;

#include <queue>
#include <cassert>

struct test_derived : private std::priority_queue<int> {
  test_derived() {
    c.push_back(1);
    assert(comp(1, 2));
  }
};

int main(int, char**) {
  test_derived t;

  return 0;
}
