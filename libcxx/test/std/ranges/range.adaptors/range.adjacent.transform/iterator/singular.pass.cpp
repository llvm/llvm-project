//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-exceptions

// If the invocation of any non-const member function of `iterator` exits via an
// exception, the iterator acquires a singular value.

#include <ranges>

#include <tuple>

#include "../helpers.h"
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

template <std::size_t N, class Fn, class Validator>
void test() {
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};
  Validator validator{};
  {
    // adjacent_transform_view iterator should be able to be destroyed after member function throws
    auto v  = ThrowOnIncrementView{buffer} | std::views::adjacent_transform<N>(Fn{});
    auto it = v.begin();
    ++it;
    try {
      --it;
      assert(false); // should not be reached as the above expression should throw.
    } catch (int e) {
      assert(e == 5);
    }
  }

  {
    // adjacent_transform_view iterator should be able to be assigned after member function throws
    auto v  = ThrowOnIncrementView{buffer} | std::views::adjacent_transform<N>(Fn{});
    auto it = v.begin();
    ++it;
    try {
      --it;
      assert(false); // should not be reached as the above expression should throw.
    } catch (int e) {
      assert(e == 5);
    }
    it = v.begin();
    validator(buffer, *it, 0);
  }
}

template <std::size_t N>
constexpr void test() {
  test<N, MakeTuple, ValidateTupleFromIndex<N>>();
  test<N, Tie, ValidateTieFromIndex<N>>();
  test<N, GetFirst, ValidateGetFirstFromIndex<N>>();
  test<N, Multiply, ValidateMultiplyFromIndex<N>>();
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
