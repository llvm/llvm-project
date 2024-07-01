//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <generator>

// template<class Ref, class V = void, class Allocator = void>
//   class generator;

#include <generator>

#include <cassert>
#include <cstddef>
#include <memory>
#include <memory_resource>
#include <ranges>
#include <utility>
#include <vector>

template <class T>
class stateless_allocator {
public:
  using value_type = T;

  stateless_allocator() noexcept = default;

  template <typename U>
  constexpr stateless_allocator(const stateless_allocator<U>&) noexcept {}

  [[nodiscard]] T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T))); }

  void deallocate(void* p, size_t) noexcept { ::operator delete(p); }

  template <class U>
  constexpr bool operator==(const stateless_allocator<U>&) const noexcept {
    return true;
  }
};

template <class T>
class stateful_allocator {
public:
  using value_type = T;

  stateful_allocator() noexcept = default;

  template <typename U>
  constexpr stateful_allocator(const stateful_allocator<U>&) noexcept {}

  [[nodiscard]] T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T))); }

  void deallocate(void* p, size_t) noexcept { ::operator delete(p); }

  template <class U>
  constexpr bool operator==(const stateful_allocator<U>& other) const noexcept {
    return state_ == other.state_;
  }

private:
  int state_ = 0;
};

template <class Allocator>
bool test_with_allocator() {
  std::vector<int> expected_fib_vec = {0, 1, 1, 2, 3};
  {
    auto fib = []() -> std::generator<int, int, Allocator> {
      int a = 0;
      int b = 1;
      while (true) {
        co_yield std::exchange(a, std::exchange(b, a + b));
      }
    };
    assert((fib() | std::views::take(5) | std::ranges::to<std::vector>()) == expected_fib_vec);
  }

  {
    auto fib = [](std::allocator_arg_t, Allocator) -> std::generator<int, int, Allocator> {
      int a = 0;
      int b = 1;
      while (true) {
        co_yield std::exchange(a, std::exchange(b, a + b));
      }
    };
    assert((fib(std::allocator_arg, {}) | std::views::take(5) | std::ranges::to<std::vector>()) == expected_fib_vec);
  }

  {
    auto fib = [](std::allocator_arg_t, Allocator) -> std::generator<int> {
      int a = 0;
      int b = 1;
      while (true) {
        co_yield std::exchange(a, std::exchange(b, a + b));
      }
    };
    assert((fib(std::allocator_arg, {}) | std::views::take(5) | std::ranges::to<std::vector>()) == expected_fib_vec);
  }
  return true;
}

bool test() {
  test_with_allocator<std::allocator<std::byte>>();
  test_with_allocator<stateless_allocator<std::byte>>();
  test_with_allocator<stateful_allocator<std::byte>>();
  test_with_allocator<std::pmr::polymorphic_allocator<std::byte>>();

  test_with_allocator<std::allocator<float>>();
  test_with_allocator<stateless_allocator<float>>();
  test_with_allocator<stateful_allocator<float>>();
  test_with_allocator<std::pmr::polymorphic_allocator<float>>();
  return true;
};

int main() {
  test();
  return 0;
}
