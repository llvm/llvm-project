//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: no-exceptions

// <generator>
// LWG 3894: yield_value(elements_of<R, Alloc>) must allow exceptions to propagate.

#include <generator>

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <ranges>
#include <utility>

enum class Failure { Begin, End, IteratorMove, SentinelCopy, Dereference, Increment };
struct Error {};

struct Iterator {
  using value_type      = int;
  using difference_type = std::ptrdiff_t;
  int* ptr;
  Failure failure;
  Iterator(int* p, Failure f) : ptr(p), failure(f) {}
  Iterator(Iterator&& other) : ptr(other.ptr), failure(other.failure) {
    if (failure == Failure::IteratorMove)
      throw Error{};
  }
  Iterator& operator=(Iterator&&) = default;
  int& operator*() const {
    if (failure == Failure::Dereference)
      throw Error{};
    return *ptr;
  }
  Iterator& operator++() {
    if (failure == Failure::Increment)
      throw Error{};
    ++ptr;
    return *this;
  }
  void operator++(int) { ++*this; }
};

struct Sentinel {
  int* ptr        = nullptr;
  Failure failure = Failure::Begin;
  Sentinel()      = default;
  Sentinel(int* p, Failure f) : ptr(p), failure(f) {}
  Sentinel(const Sentinel& other) : ptr(other.ptr), failure(other.failure) {
    if (failure == Failure::SentinelCopy)
      throw Error{};
  }
  Sentinel& operator=(const Sentinel&) = default;
  friend bool operator==(const Iterator& i, const Sentinel& s) { return i.ptr == s.ptr; }
};

struct Range {
  int value = 42;
  Failure failure;
  Iterator begin() {
    if (failure == Failure::Begin)
      throw Error{};
    return Iterator(&value, failure);
  }
  Sentinel end() {
    if (failure == Failure::End)
      throw Error{};
    return Sentinel(&value + 1, failure);
  }
};
static_assert(std::ranges::input_range<Range>);

using Promise = std::generator<const int&>::promise_type;
static_assert(!noexcept(std::declval<Promise&>().yield_value(std::ranges::elements_of(std::declval<Range&>()))));

std::generator<const int&> generate(Failure failure, bool& caught) {
  Range range{42, failure};
  try {
    co_yield std::ranges::elements_of(range);
    assert(false);
  } catch (const Error&) {
    caught = true;
  }
  co_yield 99; // The parent can continue after a failure to initialize or iterate the nested range.
}

int main(int, char**) {
  for (Failure failure :
       {Failure::Begin,
        Failure::End,
        Failure::IteratorMove,
        Failure::SentinelCopy,
        Failure::Dereference,
        Failure::Increment}) {
    bool caught = false;
    auto g      = generate(failure, caught);
    auto i      = g.begin();
    if (failure == Failure::Increment) {
      assert(*i == 42 && !caught);
      ++i;
    }
    assert(*i == 99 && caught);
    ++i;
    assert(i == g.end());
  }

  return 0;
}
