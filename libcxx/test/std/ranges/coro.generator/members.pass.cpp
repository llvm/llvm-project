//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <generator>
// generator(generator&&);
// generator& operator=(generator);
// ~generator();
// iterator begin();
// default_sentinel_t end() const noexcept;

#include <generator>

#include <cassert>
#include <concepts>
#include <iterator>
#include <memory>
#include <utility>

std::generator<int> generate(std::shared_ptr<int> lifetime, int& resumes) {
  ++resumes;
  co_yield *lifetime;
  ++resumes;
  co_yield *lifetime + 1;
  ++resumes;
}

std::generator<int> empty() { co_return; }

void test_lifetime() {
  int resumes                 = 0;
  auto lifetime               = std::make_shared<int>(42);
  std::weak_ptr<int> observer = lifetime;
  {
    auto g = generate(std::move(lifetime), resumes);
    assert(resumes == 0); // initial_suspend is lazy
    assert(!observer.expired());
    static_assert(std::same_as<decltype(std::as_const(g).end()), std::default_sentinel_t>);
    static_assert(noexcept(std::as_const(g).end()));
    (void)std::as_const(g).end();
    assert(resumes == 0);
  }
  assert(observer.expired()); // Destroying an unstarted generator releases its parameters.
  assert(resumes == 0);

  lifetime = std::make_shared<int>(42);
  observer = lifetime;
  {
    auto g = generate(std::move(lifetime), resumes);
    auto i = g.begin();
    assert(resumes == 1 && *i == 42);
  }
  assert(observer.expired()); // Destroying a suspended generator releases its frame.
  assert(resumes == 1);

  auto g = empty();
  assert(g.begin() == g.end());
}

void test_move() {
  int resumes = 0;
  auto source = generate(std::make_shared<int>(10), resumes);
  auto moved  = std::move(source);
  assert(resumes == 0);
  auto i = moved.begin();
  assert(*i == 10);

  // Moving after begin preserves existing iterators.
  auto destination = std::move(moved);
  assert(*i == 10);
  ++i;
  assert(*i == 11);

  auto old_lifetime           = std::make_shared<int>(99);
  std::weak_ptr<int> observer = old_lifetime;
  auto target                 = generate(std::move(old_lifetime), resumes);
  auto old_iterator           = target.begin();
  assert(*old_iterator == 99);
  std::same_as<std::generator<int>&> decltype(auto) result = (target = std::move(destination));
  assert(&result == &target);
  assert(observer.expired()); // Assignment destroys the previous frame.
  assert(*i == 11);
  ++i;
  assert(i == target.end());

  // A moved-from generator can be assigned to and destroyed.
  source = empty();
  assert(source.begin() == source.end());
}

int main(int, char**) {
  test_lifetime();
  test_move();
  return 0;
}
