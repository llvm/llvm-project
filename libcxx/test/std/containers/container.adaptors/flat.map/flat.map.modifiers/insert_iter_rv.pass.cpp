//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>
//     iterator insert(const_iterator position, value_type&&);

#include <flat_map>
#include <cassert>
#include <deque>

#include "MoveOnly.h"
#include "min_allocator.h"
#include "../helpers.h"
#include "test_macros.h"

template <class Container, class Pair>
void do_insert_iter_rv_test() {
  using M = Container;
  using P = Pair;
  using R = typename M::iterator;
  M m;
  std::same_as<R> decltype(auto) r = m.insert(m.end(), P(2, 2));
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(r->first == 2);
  assert(r->second == 2);

  r = m.insert(m.end(), P(1, 1));
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(r->first == 1);
  assert(r->second == 1);

  r = m.insert(m.end(), P(3, 3));
  assert(r == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(r->first == 3);
  assert(r->second == 3);

  r = m.insert(m.end(), P(3, 4));
  assert(r == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(r->first == 3);
  assert(r->second == 3);
}
int main(int, char**) {
  do_insert_iter_rv_test<std::flat_map<int, MoveOnly>, std::pair<int, MoveOnly>>();
  do_insert_iter_rv_test<std::flat_map<int, MoveOnly>, std::pair<const int, MoveOnly>>();

  {
    using M =
        std::flat_map<int,
                      MoveOnly,
                      std::less<int>,
                      std::deque<int, min_allocator<int>>,
                      std::deque<MoveOnly, min_allocator<MoveOnly>>>;
    using P  = std::pair<int, MoveOnly>;
    using CP = std::pair<const int, MoveOnly>;
    do_insert_iter_rv_test<M, P>();
    do_insert_iter_rv_test<M, CP>();
  }
  {
    using M = std::flat_map<int, MoveOnly>;
    using R = typename M::iterator;
    M m;
    R r = m.insert(m.end(), {2, MoveOnly(2)});
    assert(r == m.begin());
    assert(m.size() == 1);
    assert(r->first == 2);
    assert(r->second == 2);

    r = m.insert(m.end(), {1, MoveOnly(1)});
    assert(r == m.begin());
    assert(m.size() == 2);
    assert(r->first == 1);
    assert(r->second == 1);

    r = m.insert(m.end(), {3, MoveOnly(3)});
    assert(r == std::ranges::prev(m.end()));
    assert(m.size() == 3);
    assert(r->first == 3);
    assert(r->second == 3);

    r = m.insert(m.end(), {3, MoveOnly(3)});
    assert(r == std::ranges::prev(m.end()));
    assert(m.size() == 3);
    assert(r->first == 3);
    assert(r->second == 3);
  }

  {
    auto insert_func = [](auto& m, auto key_arg, auto value_arg) {
      using FlatMap    = std::decay_t<decltype(m)>;
      using value_type = typename FlatMap::value_type;
      value_type p(std::piecewise_construct, std::tuple(key_arg), std::tuple(value_arg));
      m.insert(m.begin(), std::move(p));
    };
    test_emplace_exception_guarantee(insert_func);
  }

  return 0;
}
