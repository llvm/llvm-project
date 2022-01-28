//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges


// constexpr iterator_t<R> begin();
// constexpr sentinel_t<R> end();
// constexpr auto begin() const requires range<const R>;
// constexpr auto end() const requires range<const R>;

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>

#include "test_iterators.h"
#include "test_macros.h"

struct Base {
  constexpr int *begin() { return nullptr; }
  constexpr auto end() { return sentinel_wrapper<int*>(nullptr); }
  constexpr char *begin() const { return nullptr; }
  constexpr auto end() const { return sentinel_wrapper<char*>(nullptr); }
};
static_assert(std::same_as<std::ranges::iterator_t<Base>, int*>);
static_assert(std::same_as<std::ranges::sentinel_t<Base>, sentinel_wrapper<int*>>);
static_assert(std::same_as<std::ranges::iterator_t<const Base>, char*>);
static_assert(std::same_as<std::ranges::sentinel_t<const Base>, sentinel_wrapper<char*>>);

struct NoConst {
  int* begin();
  sentinel_wrapper<int*> end();
};

struct DecayChecker {
  int*& begin() const;
  int*& end() const;
};

template <class T>
concept HasBegin = requires (T t) {
  t.begin();
};

template <class T>
concept HasEnd = requires (T t) {
  t.end();
};

constexpr bool test()
{
  {
    using OwningView = std::ranges::owning_view<Base>;
    OwningView ov;
    std::same_as<int*> decltype(auto) b1 = static_cast<OwningView&>(ov).begin();
    std::same_as<int*> decltype(auto) b2 = static_cast<OwningView&&>(ov).begin();
    std::same_as<char*> decltype(auto) b3 = static_cast<const OwningView&>(ov).begin();
    std::same_as<char*> decltype(auto) b4 = static_cast<const OwningView&&>(ov).begin();

    std::same_as<sentinel_wrapper<int*>> decltype(auto) e1 = static_cast<OwningView&>(ov).end();
    std::same_as<sentinel_wrapper<int*>> decltype(auto) e2 = static_cast<OwningView&&>(ov).end();
    std::same_as<sentinel_wrapper<char*>> decltype(auto) e3 = static_cast<const OwningView&>(ov).end();
    std::same_as<sentinel_wrapper<char*>> decltype(auto) e4 = static_cast<const OwningView&&>(ov).end();

    assert(b1 == e1);
    assert(b2 == e2);
    assert(b3 == e3);
    assert(b4 == e4);
  }
  {
    // NoConst has non-const begin() and end(); so does the owning_view.
    using OwningView = std::ranges::owning_view<NoConst>;
    static_assert(HasBegin<OwningView&>);
    static_assert(HasBegin<OwningView&&>);
    static_assert(!HasBegin<const OwningView&>);
    static_assert(!HasBegin<const OwningView&&>);
    static_assert(HasEnd<OwningView&>);
    static_assert(HasEnd<OwningView&&>);
    static_assert(!HasEnd<const OwningView&>);
    static_assert(!HasEnd<const OwningView&&>);
  }
  {
    // DecayChecker's begin() and end() return references; make sure the owning_view decays them.
    using OwningView = std::ranges::owning_view<DecayChecker>;
    OwningView ov;
    ASSERT_SAME_TYPE(decltype(ov.begin()), int*);
    ASSERT_SAME_TYPE(decltype(ov.end()), int*);
  }
  {
    // Test an empty view.
    int a[] = {1};
    auto ov = std::ranges::owning_view(std::ranges::subrange(a, a));
    assert(ov.begin() == a);
    assert(std::as_const(ov).begin() == a);
    assert(ov.end() == a);
    assert(std::as_const(ov).end() == a);
  }
  {
    // Test a non-empty view.
    int a[] = {1};
    auto ov = std::ranges::owning_view(std::ranges::subrange(a, a+1));
    assert(ov.begin() == a);
    assert(std::as_const(ov).begin() == a);
    assert(ov.end() == a+1);
    assert(std::as_const(ov).end() == a+1);
  }
  {
    // Test a non-view.
    std::array<int, 2> a = {1, 2};
    auto ov = std::ranges::owning_view(std::move(a));
    assert(std::to_address(ov.begin()) != std::to_address(a.begin())); // because it points into the copy
    assert(std::to_address(std::as_const(ov).begin()) != std::to_address(a.begin()));
    assert(std::to_address(ov.end()) != std::to_address(a.end()));
    assert(std::to_address(std::as_const(ov).end()) != std::to_address(a.end()));
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
