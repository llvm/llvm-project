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
#include <string>
#include <type_traits>
#include <vector>
#include "test_iterators.h"
#include "test_macros.h"

// test concept constraints

template <typename T>
concept WellFormedView = requires(T& a) { std::views::concat(a); };

struct X {};
struct Y {};

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
  constexpr InputRange(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Iterator begin() { return Iterator(begin_); }
  constexpr Sentinel end() { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

struct RefOnlyRange1 : std::ranges::view_base {
  X* begin() const;
  X* end() const;
};

struct RefOnlyRange2 : std::ranges::view_base {
  Y* begin() const;
  Y* end() const;
};

namespace std {
template <template <class> class TQual, template <class> class UQual>
struct basic_common_reference< X, Y, TQual, UQual> {
  using type = X;
};

template <template <class> class TQual, template <class> class UQual>
struct basic_common_reference< Y, X, TQual, UQual> {
  using type = X;
};
} // namespace std

struct R1 : std::ranges::view_base {
  int* first = nullptr;
  int* last  = nullptr;

  R1() = default;
  R1(int* f, int* l) : first(f), last(l) {}

  struct iterator {
    using value_type       = int;
    using difference_type  = std::ptrdiff_t;
    using iterator_concept = std::forward_iterator_tag;

    int* p = nullptr;

    int& operator*() const { return *p; }
    iterator& operator++() {
      ++p;
      return *this;
    }
    iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    friend bool operator==(const iterator& x, const iterator& y) { return x.p == y.p; }

    friend X iter_move(const iterator& it) {
      (void)it;
      return X{};
    }
  };

  iterator begin() const { return iterator{first}; }
  iterator end() const { return iterator{last}; }
};

struct R2 : std::ranges::view_base {
  int* first = nullptr;
  int* last  = nullptr;

  R2() = default;
  R2(int* f, int* l) : first(f), last(l) {}

  struct iterator {
    using value_type       = int;
    using difference_type  = std::ptrdiff_t;
    using iterator_concept = std::forward_iterator_tag;

    int* p = nullptr;

    int& operator*() const { return *p; }
    iterator& operator++() {
      ++p;
      return *this;
    }
    iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    friend bool operator==(const iterator& x, const iterator& y) { return x.p == y.p; }

    friend Y iter_move(const iterator& it) {
      (void)it;
      return Y{};
    }
  };

  iterator begin() const { return iterator{first}; }
  iterator end() const { return iterator{last}; }
};

struct MoveOnlyIterator {
  using It = int*;

  It it_;

  using iterator_category = std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = std::ptrdiff_t;
  using reference         = int&;

  constexpr explicit MoveOnlyIterator(It it) : it_(it) {}
  MoveOnlyIterator(MoveOnlyIterator&&)                 = default;
  MoveOnlyIterator& operator=(MoveOnlyIterator&&)      = default;
  MoveOnlyIterator(const MoveOnlyIterator&)            = delete;
  MoveOnlyIterator& operator=(const MoveOnlyIterator&) = delete;

  constexpr reference operator*() const { return *it_; }

  constexpr MoveOnlyIterator& operator++() {
    ++it_;
    return *this;
  }
  constexpr MoveOnlyIterator operator++(int) { return MoveOnlyIterator(it_++); }

  friend constexpr bool operator==(const MoveOnlyIterator& x, const MoveOnlyIterator& y) { return x.it_ == y.it_; }
  friend constexpr bool operator!=(const MoveOnlyIterator& x, const MoveOnlyIterator& y) { return x.it_ != y.it_; }
};

struct MoveOnlyView : std::ranges::view_base {
  int* b;
  int* e;
  constexpr MoveOnlyView() = default;
  constexpr MoveOnlyView(int* b, int* e) : b(b), e(e) {}
  MoveOnlyView(const MoveOnlyView&) = delete;
  constexpr MoveOnlyView(MoveOnlyView&& other) : b(other.b), e(other.e) {}
  MoveOnlyView& operator=(const MoveOnlyView&) = delete;
  constexpr MoveOnlyView& operator=(MoveOnlyView&& other) {
    b = other.b;
    e = other.e;
    return *this;
  }

  constexpr auto begin() const { return MoveOnlyIterator{b}; }
  constexpr auto end() const { return MoveOnlyIterator{e}; }
};

template <typename... Ts>
concept ConcatViewConstraintsPass = requires(Ts&&... a) { std::views::concat(a...); };

int main(int, char**) {
  // rejects when it is an output range
  {
    std::vector<int> v{1, 2, 3};
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
      // concat-value-t is ill-formed but concat-reference-t is valid
      static_assert(!ConcatViewConstraintsPass<RefOnlyRange1, RefOnlyRange2>);
    }

    {
      // concat-rvalue-reference-t is ill-formed
      static_assert(!ConcatViewConstraintsPass<R1, R2>);
    }

    {
      // concat-indirectly-readable is ill-formed
      static_assert(!ConcatViewConstraintsPass<BadView, BadView>);
    }

    {
      // concatable fails when there is a MoveOnly& and MoveOnly
      // Let Fs be a pack containing MoveOnly& and MoveOnly
      // common_reference_with<concat_reference_t<Fs...>, concat_value_t<Fs...>> fails
      static_assert(!ConcatViewConstraintsPass<MoveOnlyView&, MoveOnlyView>);
    }
  }

  return 0;
}
