//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// Note: atomic<weak_ptr<T>> intentionally has no constexpr atomic(nullptr_t) constructor,
// unlike atomic<shared_ptr<T>>. See [util.smartptr.atomic.weak].

// using value_type = weak_ptr<T>;
// static constexpr bool is_always_lock_free = false;
// atomic() noexcept;
// atomic(weak_ptr<T>) noexcept;
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
  using A = std::atomic<std::weak_ptr<T>>;

  static_assert(std::same_as<typename A::value_type, std::weak_ptr<T>>);
  static_assert(!A::is_always_lock_free);

  static_assert(!std::is_copy_constructible_v<A>);
  static_assert(!std::is_copy_assignable_v<A>);

  static_assert(std::is_nothrow_default_constructible_v<A>);

  {
    A a;
    assert(a.load().expired());
  }

  {
    auto sp             = SpValues<T>::state_a();
    std::weak_ptr<T> wp = sp;
    A a((std::weak_ptr<T>(wp)));
    auto locked = a.load().lock();
    assert(locked && *locked == *sp);
  }
}

template <class T>
struct TestTraitsWeak {
  void operator()() const { traits<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestTraitsWeak>();
  return 0;
}
