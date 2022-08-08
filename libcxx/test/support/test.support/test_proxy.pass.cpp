//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

#include "MoveOnly.h"
#include "test_iterators.h"

#include <cassert>

constexpr void testProxy() {
  // constructor value
  {
    Proxy<int> p{5};
    assert(p.data == 5);
  }

  // constructor reference
  {
    int i = 5;
    Proxy<int&> p{i};
    assert(&p.data == &i);
  }

  // constructor conversion
  {
    int i = 5;
    Proxy<int&> p1{i};
    Proxy<int> p2 = p1;
    assert(p2.data == 5);

    Proxy<int&> p3{p2};
    assert(&(p3.data) == &(p2.data));

    MoveOnly m1{8};
    Proxy<MoveOnly&&> p4 = std::move(m1);

    Proxy<MoveOnly> p5 = std::move(p4);
    assert(p5.data.get() == 8);
  }

  // assignment
  {
    Proxy<int> p1{5};
    Proxy<int> p2{6};
    p1 = p2;
    assert(p1.data == 6);

    MoveOnly m1{8};
    Proxy<MoveOnly&&> p3 = std::move(m1);
    Proxy<MoveOnly> p4{MoveOnly{9}};
    p4 = std::move(p3);
    assert(p4.data.get() == 8);

    // `T` is a reference type.
    int i = 5, j = 6, k = 7, x = 8;
    Proxy<int&> p5{i};
    // `Other` is a prvalue.
    p5 = Proxy<int&>{j};
    assert(p5.data == 6);
    // `Other` is a const lvalue.
    const Proxy<int&> p_ref{k};
    p5 = p_ref;
    assert(p5.data == 7);
    // `Other` is an xvalue.
    Proxy<int&> px{x};
    p5 = std::move(px);
    assert(p5.data == 8);
  }

  // const assignment
  {
    int i = 5;
    int j = 6;
    const Proxy<int&> p1{i};
    const Proxy<int&> p2{j};
    p1 = p2;
    assert(i == 6);

    MoveOnly m1{8};
    MoveOnly m2{9};
    Proxy<MoveOnly&&> p3       = std::move(m1);
    const Proxy<MoveOnly&&> p4 = std::move(m2);
    p4                         = std::move(p3);
    assert(p4.data.get() == 8);
  }

  // compare
  {
    Proxy<int> p1{5};
    Proxy<int> p2{6};
    assert(p1 != p2);
    assert(p1 < p2);

    // Comparing `T` and `T&`.
    int i = 5, j = 6;
    Proxy<int&> p_ref{i};
    Proxy<const int&> p_cref{j};
    assert(p1 == p_ref);
    assert(p2 == p_cref);
    assert(p_ref == p1);
    assert(p_cref == p2);
    assert(p_ref == p_ref);
    assert(p_cref == p_cref);
    assert(p_ref != p_cref);
  }
}

static_assert(std::input_iterator<ProxyIterator<cpp20_input_iterator<int*>>>);
static_assert(!std::forward_iterator<ProxyIterator<cpp20_input_iterator<int*>>>);

static_assert(std::forward_iterator<ProxyIterator<forward_iterator<int*>>>);
static_assert(!std::bidirectional_iterator<ProxyIterator<forward_iterator<int*>>>);

static_assert(std::bidirectional_iterator<ProxyIterator<bidirectional_iterator<int*>>>);
static_assert(!std::random_access_iterator<ProxyIterator<bidirectional_iterator<int*>>>);

static_assert(std::random_access_iterator<ProxyIterator<random_access_iterator<int*>>>);
static_assert(!std::contiguous_iterator<ProxyIterator<random_access_iterator<int*>>>);

static_assert(std::random_access_iterator<ProxyIterator<contiguous_iterator<int*>>>);
static_assert(!std::contiguous_iterator<ProxyIterator<contiguous_iterator<int*>>>);

template <class Iter>
constexpr void testInputIteratorOperation() {
  int data[] = {1, 2};
  ProxyIterator<Iter> iter{Iter{data}};
  sentinel_wrapper<ProxyIterator<Iter>> sent{ProxyIterator<Iter>{Iter{data + 2}}};

  std::same_as<Proxy<int&>> decltype(auto) result = *iter;
  assert(result.data == 1);
  auto& iter2 = ++iter;
  static_assert(std::is_same_v<decltype(++iter), ProxyIterator<Iter>&>);
  assert(&iter2 == &iter);
  assert((*iter).data == 2);
  ++iter;
  assert(iter == sent);
}

template <class Iter>
constexpr void testForwardIteratorOperation() {
  int data[] = {1, 2};
  ProxyIterator<Iter> iter{Iter{data}};

  std::same_as<ProxyIterator<Iter>> decltype(auto) it2 = iter++;
  assert((*it2).data == 1);
  assert((*iter).data == 2);
}

template <class Iter>
constexpr void testBidirectionalIteratorOperation() {
  int data[] = {1, 2};
  ProxyIterator<Iter> iter{Iter{data}};
  ++iter;
  assert((*iter).data == 2);

  auto& iter2 = --iter;
  static_assert(std::is_same_v<decltype(--iter), ProxyIterator<Iter>&>);
  assert(&iter2 == &iter);
  assert((*iter).data == 1);
  ++iter;

  std::same_as<ProxyIterator<Iter>> decltype(auto) iter3 = iter--;
  assert((*iter).data == 1);
  assert((*iter3).data == 2);
}

template <class Iter>
constexpr void testRandomAccessIteratorOperation() {
  int data[] = {1, 2, 3, 4, 5};
  ProxyIterator<Iter> iter{Iter{data}};

  auto& iter2 = iter += 2;
  static_assert(std::is_same_v<decltype(iter += 2), ProxyIterator<Iter>&>);
  assert(&iter2 == &iter);
  assert((*iter).data == 3);

  auto& iter3 = iter -= 1;
  static_assert(std::is_same_v<decltype(iter -= 1), ProxyIterator<Iter>&>);
  assert(&iter3 == &iter);
  assert((*iter).data == 2);

  std::same_as<Proxy<int&>> decltype(auto) r = iter[2];
  assert(r.data == 4);

  std::same_as<ProxyIterator<Iter>> decltype(auto) iter4 = iter - 1;
  assert((*iter4).data == 1);

  std::same_as<ProxyIterator<Iter>> decltype(auto) iter5 = iter4 + 2;
  assert((*iter5).data == 3);

  std::same_as<ProxyIterator<Iter>> decltype(auto) iter6 = 3 + iter4;
  assert((*iter6).data == 4);

  std::same_as<std::iter_difference_t<Iter>> decltype(auto) n = iter6 - iter5;
  assert(n == 1);

  assert(iter4 < iter5);
  assert(iter3 <= iter5);
  assert(iter5 > iter4);
  assert(iter6 >= iter4);
}

constexpr void testProxyIterator() {
  // input iterator operations
  {
    testInputIteratorOperation<cpp20_input_iterator<int*>>();
    testInputIteratorOperation<forward_iterator<int*>>();
    testInputIteratorOperation<bidirectional_iterator<int*>>();
    testInputIteratorOperation<random_access_iterator<int*>>();
    testInputIteratorOperation<contiguous_iterator<int*>>();
  }

  // forward iterator operations
  {
    testForwardIteratorOperation<forward_iterator<int*>>();
    testForwardIteratorOperation<bidirectional_iterator<int*>>();
    testForwardIteratorOperation<random_access_iterator<int*>>();
    testForwardIteratorOperation<contiguous_iterator<int*>>();
  }

  // bidirectional iterator operations
  {
    testBidirectionalIteratorOperation<bidirectional_iterator<int*>>();
    testBidirectionalIteratorOperation<random_access_iterator<int*>>();
    testBidirectionalIteratorOperation<contiguous_iterator<int*>>();
  }

  // random access iterator operations
  {
    testRandomAccessIteratorOperation<random_access_iterator<int*>>();
    testRandomAccessIteratorOperation<contiguous_iterator<int*>>();
  }
}

constexpr void testProxyRange() {
  int data[] = {3, 4, 5};
  ProxyRange r{data};
  std::same_as<ProxyIterator<int*>> decltype(auto) it = std::ranges::begin(r);
  assert((*it).data == 3);
  it += 3;
  assert(it == std::ranges::end(r));
}

template <class Iter>
concept StdMoveWorks = requires(std::iter_value_t<Iter> val, Iter iter) { val = std::move(*iter); };

static_assert(StdMoveWorks<MoveOnly*>);
static_assert(!StdMoveWorks<ProxyIterator<MoveOnly*>>);

// although this "works" but it actually creates a copy instead of move
static_assert(StdMoveWorks<ProxyIterator<int*>>);

using std::swap;

template <class Iter>
concept SwapWorks = requires(Iter iter1, Iter iter2) { swap(*iter1, *iter2); };

static_assert(SwapWorks<int*>);
static_assert(!SwapWorks<ProxyIterator<int*>>);

constexpr bool test() {
  testProxy();
  testProxyIterator();
  testProxyRange();

  // iter_move
  {
    MoveOnly data[] = {5, 6, 7};
    ProxyRange r{data};
    auto it                               = r.begin();
    std::iter_value_t<decltype(it)> moved = std::ranges::iter_move(it);
    assert(moved.data.get() == 5);
  }

  // iter_swap
  {
    MoveOnly data[] = {5, 6, 7};
    ProxyRange r{data};
    auto it1 = r.begin();
    auto it2 = it1 + 2;
    std::ranges::iter_swap(it1, it2);
    assert(data[0].get() == 7);
    assert(data[2].get() == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
