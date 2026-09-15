//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr decltype(auto) operator*() const

#include <cassert>
#include <ranges>

#include "../types.h"
#include "test_iterators.h"

constexpr bool test() {
  {
    // Dereference with stride 1.
    int arr[]  = {10, 20, 30};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 3), 1);
    auto it    = sv.begin();

    assert(*it == 10);
    ++it;
    assert(*it == 20);
    ++it;
    assert(*it == 30);
  }
  {
    // Dereference with stride 2.
    int arr[]  = {10, 20, 30, 40, 50};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 5), 2);
    auto it    = sv.begin();

    assert(*it == 10);
    ++it;
    assert(*it == 30);
    ++it;
    assert(*it == 50);
  }
  {
    // Dereference with stride larger than range.
    int arr[]  = {42, 99};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 2), 5);
    auto it    = sv.begin();

    assert(*it == 42);
  }
  {
    // Dereference returns a reference that can be assigned through.
    int arr[]  = {1, 2, 3, 4, 5};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 5), 2);
    auto it    = sv.begin();

    *it = 100;
    assert(arr[0] == 100);
    ++it;
    *it = 200;
    assert(arr[2] == 200);
  }
  {
    // Dereference on a forward range with stride 3.
    int arr[]  = {5, 10, 15, 20, 25, 30};
    using Base = BasicTestView<forward_iterator<int*>, forward_iterator<int*>>;
    auto sv    = std::ranges::stride_view(Base(forward_iterator(arr), forward_iterator(arr + 6)), 3);
    auto it    = sv.begin();
    assert(*it == 5);
    ++it;
    assert(*it == 20);
  }
  {
    // Return type is a reference, not a value.
    int arr[]  = {1, 2, 3};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(arr, arr + 3), 1);
    auto it    = sv.begin();
    static_assert(std::is_same_v<decltype(*it), int&>);
  }
  {
    // Dereference through a const-qualified iterator.
    int arr[]      = {10, 20, 30};
    using Base     = BasicTestView<int*, int*>;
    auto sv        = std::ranges::stride_view(Base(arr, arr + 3), 2);
    const auto cit = sv.begin();
    assert(*cit == 10);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
