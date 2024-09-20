//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// iterator insert(const_iterator position, const value_type& v);

#include <flat_map>
#include <cassert>
#include <functional>
#include <deque>

#include "test_macros.h"
#include "min_allocator.h"

template <class Container>
void do_insert_iter_cv_test() {
  using M  = Container;
  using R  = typename M::iterator;
  using VT = typename M::value_type;

  M m;
  const VT v1(2, 2.5);
  std::same_as<R> decltype(auto) r = m.insert(m.end(), v1);
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(r->first == 2);
  assert(r->second == 2.5);

  const VT v2(1, 1.5);
  r = m.insert(m.end(), v2);
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(r->first == 1);
  assert(r->second == 1.5);

  const VT v3(3, 3.5);
  r = m.insert(m.end(), v3);
  assert(r == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(r->first == 3);
  assert(r->second == 3.5);

  const VT v4(3, 4.5);
  r = m.insert(m.end(), v4);
  assert(r == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(r->first == 3);
  assert(r->second == 3.5);
}

int main(int, char**) {
  do_insert_iter_cv_test<std::flat_map<int, double> >();
  {
    using M =
        std::flat_map<int,
                      double,
                      std::less<int>,
                      std::deque<int, min_allocator<int>>,
                      std::deque<double, min_allocator<double>>>;
    do_insert_iter_cv_test<M>();
  }

  return 0;
}
