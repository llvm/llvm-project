//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <cassert>
#include <list>
#include <ranges>
#include <type_traits>
#include <vector>
#include "test_iterators.h"
#include "test_macros.h"

// test concept constraints

template<typename T>
concept WellFormedView = requires(T& a) {
  std::views::concat(a);
};

struct X {};

struct BadIter {
  using value_type = int;
  int* p;

  BadIter() = default;
  explicit BadIter(int* q) : p(q) {}

  int& operator*() const { return *p; }
  BadIter& operator++() {
    ++p;
    return *this;
  }
  void operator++(int) { ++p; }

  friend bool operator==(const BadIter& a, const BadIter& b) { return a.p == b.p; }
  friend bool operator!=(const BadIter& a, const BadIter& b) { return !(a == b); }

  friend X iter_move(const BadIter&) { return X{}; }
};

struct BadView : std::ranges::view_base {
  int buf_[1] = {0};
  BadIter begin() const { return BadIter(const_cast<int*>(buf_)); }
  BadIter end() const { return BadIter(const_cast<int*>(buf_ + 1)); }
};

struct InputRange {
  using Iterator = cpp17_input_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr InputRange(int* b, int *e): begin_(b), end_(e) {}
  constexpr Iterator begin() { return Iterator(begin_); }
  constexpr Sentinel end() { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

template <typename... Ts>
concept ConcatViewConstraintsPass = requires(Ts&&... a) { std::views::concat(a...); };

int main(int, char**) {

  // rejects when it is an output range
  {
    std::vector<int> v{1,2,3};
    static_assert(!WellFormedView<decltype(std::views::counted(std::back_inserter(v), 3))>);
  }

  // input range
  {
    static_assert(WellFormedView<InputRange>);
  }

  // bidirectional range
  {
    static_assert(WellFormedView<std::list<int>>);
  }

  // random access range
  {
    static_assert(WellFormedView<std::vector<int>>);
  }

  {
    // LWG 4082
    std::vector<int> v{1, 2, 3};
    auto r = std::views::counted(std::back_inserter(v), 3);
    //auto c = std::views::concat(r);
    static_assert(!ConcatViewConstraintsPass<decltype(r)>);
  }

  {
    // input is a view and has 0 size
    static_assert(!ConcatViewConstraintsPass<>);
  }

  {
    // input is a view and has at least an element
    static_assert(ConcatViewConstraintsPass<std::vector<int>>);
  }

  {
    // inputs are non-concatable
    static_assert(!ConcatViewConstraintsPass<std::vector<int>, std::vector<std::string>>);
  }

  {
    // test concept concatable
    {
      // concat-reference-t is ill-formed
      // no common reference between int and string
      static_assert(!ConcatViewConstraintsPass<std::array<int, 1>, std::array<std::string, 1>>);
    }

    {
      // concat-indirectly-readable is ill-formed
      //   since iter_move of BadIter returns an unrelated type
      static_assert(!ConcatViewConstraintsPass<BadView, BadView>);
    }
  }

  return 0;
}
