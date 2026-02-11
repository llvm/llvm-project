//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-exceptions

// [range.adjacent.iterator#2] If the invocation of any non-const member function of `iterator` exits via an
// exception, the iterator acquires a singular value.

#include <ranges>

#include <tuple>

#include "../../range_adaptor_types.h"

struct ThrowOnDecrementIterator {
  int* it_;

  using value_type      = int;
  using difference_type = std::intptr_t;

  ThrowOnDecrementIterator() = default;
  explicit ThrowOnDecrementIterator(int* it) : it_(it) {}

  ThrowOnDecrementIterator& operator++() {
    ++it_;
    return *this;
  }
  ThrowOnDecrementIterator operator++(int) {
    auto tmp = *this;
    ++it_;
    return tmp;
  }

  ThrowOnDecrementIterator& operator--() { throw 5; }
  ThrowOnDecrementIterator operator--(int) { throw 5; }

  int& operator*() const { return *it_; }

  friend bool operator==(ThrowOnDecrementIterator const&, ThrowOnDecrementIterator const&) = default;
};

struct ThrowOnIncrementView : IntBufferView {
  ThrowOnDecrementIterator begin() const { return ThrowOnDecrementIterator{buffer_}; }
  ThrowOnDecrementIterator end() const { return ThrowOnDecrementIterator{buffer_ + size_}; }
};

template <std::size_t N>
void test() {
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    // adjacent_view iterator should be able to be destroyed after member function throws
    auto v  = ThrowOnIncrementView{buffer} | std::views::adjacent<N>;
    auto it = v.begin();
    ++it;
    try {
      --it;
      assert(false); // should not be reached as the above expression should throw.
    } catch (int e) {
      assert(e == 5);
    }
    // destroy the iterator
  }

  {
    // adjacent_view iterator should be assignable after member function throws
    auto v  = ThrowOnIncrementView{buffer} | std::views::adjacent<N>;
    auto it = v.begin();
    ++it;
    try {
      --it;
      assert(false); // should not be reached as the above expression should throw.
    } catch (int e) {
      assert(e == 5);
    }
    it         = v.begin();
    auto tuple = *it;
    assert(std::get<0>(tuple) == buffer[0]);
    if constexpr (N >= 2)
      assert(std::get<1>(tuple) == buffer[1]);
    if constexpr (N >= 3)
      assert(std::get<2>(tuple) == buffer[2]);
    if constexpr (N >= 4)
      assert(std::get<3>(tuple) == buffer[3]);
    if constexpr (N >= 5)
      assert(std::get<4>(tuple) == buffer[4]);
  }
}

void test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();
}

int main(int, char**) {
  test();

  return 0;
}
