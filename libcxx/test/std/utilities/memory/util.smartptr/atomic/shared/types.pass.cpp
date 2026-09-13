//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// using value_type = shared_ptr<T>;
// static constexpr bool is_always_lock_free = false;
// atomic() noexcept;
// constexpr atomic(nullptr_t) noexcept;
// atomic(shared_ptr<T>) noexcept;
// atomic(const atomic&) = delete;
// atomic& operator=(const atomic&) = delete;

#include <atomic>
#include <cassert>
#include <concepts>
#include <memory>
#include <type_traits>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void traits() {
  using A = std::atomic<std::shared_ptr<T>>;

  static_assert(std::same_as<typename A::value_type, std::shared_ptr<T>>);
  static_assert(!A::is_always_lock_free);

  static_assert(!std::is_copy_constructible_v<A>);
  static_assert(!std::is_copy_assignable_v<A>);

  static_assert(std::is_nothrow_default_constructible_v<A>);
  static_assert(std::is_nothrow_constructible_v<A, std::nullptr_t>);

  {
    A a;
    assert(!a.load());
  }

  {
    auto p = SpValues<T>::state_a();
    A a((std::shared_ptr<T>(p)));
    assert(a.load().get() == p.get());
    assert(*a.load() == *p);
  }
}

template <class T>
struct TestTraitsShared {
  void operator()() const { traits<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestTraitsShared>();

  {
    static_assert(std::is_nothrow_constructible_v<std::atomic<std::shared_ptr<int>>, std::nullptr_t>);
    std::atomic<std::shared_ptr<int>> a(nullptr);
    assert(!a.load());
  }

  return 0;
}
