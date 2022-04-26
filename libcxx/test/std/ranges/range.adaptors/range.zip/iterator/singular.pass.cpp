//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-has-no-incomplete-ranges, no-exceptions

// If the invocation of any non-const member function of `iterator` exits via an
// exception, the iterator acquires a singular value.

#include <ranges>

#include <tuple>

#include "../types.h"

struct ThrowOnIncrementIterator {
  int* it_;

  using value_type = int;
  using difference_type = std::intptr_t;
  using iterator_concept = std::input_iterator_tag;

  ThrowOnIncrementIterator() = default;
  explicit ThrowOnIncrementIterator(int* it) : it_(it) {}

  ThrowOnIncrementIterator& operator++() {
    ++it_;
    throw 5;
    return *this;
  }
  void operator++(int) { ++it_; }

  int& operator*() const { return *it_; }

  friend bool operator==(ThrowOnIncrementIterator const&, ThrowOnIncrementIterator const&) = default;
};

struct ThrowOnIncrementView : IntBufferView {
  ThrowOnIncrementIterator begin() const { return ThrowOnIncrementIterator{buffer_}; }
  ThrowOnIncrementIterator end() const { return ThrowOnIncrementIterator{buffer_ + size_}; }
};

// Cannot run the test at compile time because it is not allowed to throw exceptions
void test() {
  int buffer[] = {1, 2, 3};
  {
    // zip iterator should be able to be destroyed after member function throws
    std::ranges::zip_view v{ThrowOnIncrementView{buffer}};
    auto it = v.begin();
    try {
      ++it;
      assert(false); // should not be reached as the above expression should throw.
    } catch (int e) {
      assert(e == 5);
    }
  }

  {
    // zip iterator should be able to be assigned after member function throws
    std::ranges::zip_view v{ThrowOnIncrementView{buffer}};
    auto it = v.begin();
    try {
      ++it;
      assert(false); // should not be reached as the above expression should throw.
    } catch (int e) {
      assert(e == 5);
    }
    it = v.begin();
    auto [x] = *it;
    assert(x == 1);
  }
}

int main(int, char**) {
  test();

  return 0;
}
