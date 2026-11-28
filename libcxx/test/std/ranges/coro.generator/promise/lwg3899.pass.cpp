//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <generator>
// LWG 3899: co_yielding elements of an lvalue generator uses the generator overload.

#include <generator>

#include <cassert>
#include <initializer_list>
#include <memory>
#include <ranges>
#include <utility>

// A generator delegation does not use the allocator in elements_of.
// Instantiating the general range overload with this type would be ill-formed.
struct UnusedAllocator {};

using Child = std::generator<int&&, int, std::allocator<int>>;

Child child(std::shared_ptr<int> lifetime) { co_yield *lifetime; }

std::generator<int> parent(bool lvalue, bool& destroyed) {
  auto lifetime               = std::make_shared<int>(42);
  std::weak_ptr<int> observer = lifetime;
  auto g                      = child(std::move(lifetime));
  using P                     = std::generator<int>::promise_type;
  static_assert(noexcept(std::declval<P&>().yield_value(std::ranges::elements_of(g, UnusedAllocator{}))));
  static_assert(noexcept(std::declval<P&>().yield_value(std::ranges::elements_of(std::move(g), UnusedAllocator{}))));
  if (lvalue)
    co_yield std::ranges::elements_of(g, UnusedAllocator{});
  else
    co_yield std::ranges::elements_of(std::move(g), UnusedAllocator{});
  // Parameters live until the coroutine frame is destroyed, not merely until its body returns.
  assert(observer.expired());
  destroyed = true;
  co_yield 43;
}

int main(int, char**) {
  for (bool lvalue : {false, true}) {
    bool destroyed = false;
    auto g         = parent(lvalue, destroyed);
    auto i         = g.begin();
    assert(*i == 42 && !destroyed);
    ++i;
    assert(*i == 43 && destroyed);
    ++i;
    assert(i == g.end());
  }
  return 0;
}
