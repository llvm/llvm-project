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
#include <memory>
#include <ranges>
#include <utility>

enum class Failure { Begin, End, IteratorMove, SentinelCopy, Dereference, Increment };
struct Error {};

template <Failure failure>
struct Iterator {
  using value_type      = int;
  using difference_type = std::ptrdiff_t;
  int* ptr;
  explicit Iterator(int* p) : ptr(p) {}
  Iterator(Iterator&& other) noexcept(failure != Failure::IteratorMove) : ptr(other.ptr) {
    if constexpr (failure == Failure::IteratorMove)
      throw Error{};
  }
  Iterator& operator=(Iterator&&) = default;
  int& operator*() const noexcept(failure != Failure::Dereference) {
    if constexpr (failure == Failure::Dereference)
      throw Error{};
    return *ptr;
  }
  Iterator& operator++() noexcept(failure != Failure::Increment) {
    if constexpr (failure == Failure::Increment)
      throw Error{};
    ++ptr;
    return *this;
  }
  void operator++(int) { ++*this; }
};

template <Failure failure>
struct Sentinel {
  int* ptr   = nullptr;
  Sentinel() = default;
  explicit Sentinel(int* p) : ptr(p) {}
  Sentinel(const Sentinel& other) noexcept(failure != Failure::SentinelCopy) : ptr(other.ptr) {
    if constexpr (failure == Failure::SentinelCopy)
      throw Error{};
  }
  Sentinel& operator=(const Sentinel&) = default;
  friend bool operator==(const Iterator<failure>& i, const Sentinel& s) noexcept { return i.ptr == s.ptr; }
};

template <Failure failure>
struct Range {
  int value = 42;
  Iterator<failure> begin() {
    if constexpr (failure == Failure::Begin)
      throw Error{};
    return Iterator<failure>(&value);
  }
  Sentinel<failure> end() {
    if constexpr (failure == Failure::End)
      throw Error{};
    return Sentinel<failure>(&value + 1);
  }
};

template <Failure failure>
std::generator<const int&> generate(bool& caught) {
  Range<failure> range;
  try {
    co_yield std::ranges::elements_of(range);
    assert(false);
  } catch (const Error&) {
    caught = true;
  }
  co_yield 99; // The parent can continue after a failure to initialize or iterate the nested range.
}

template <Failure failure>
void test() {
  using R       = Range<failure>;
  using Promise = std::generator<const int&>::promise_type;
  static_assert(std::ranges::input_range<R>);
  static_assert(!noexcept(std::declval<Promise&>().yield_value(std::ranges::elements_of(std::declval<R&>()))));
  bool caught = false;
  auto g      = generate<failure>(caught);
  auto i      = g.begin();
  if constexpr (failure == Failure::Increment) {
    assert(*i == 42 && !caught);
    ++i;
  }
  assert(*i == 99 && caught);
  ++i;
  assert(i == g.end());
}

int main(int, char**) {
  test<Failure::Begin>();
  test<Failure::End>();
  test<Failure::IteratorMove>();
  test<Failure::SentinelCopy>();
  test<Failure::Dereference>();
  test<Failure::Increment>();
  return 0;
}
