//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// UNSUPPORTED: c++03, c++11, c++17

#include <atomic>

struct arr {
  int x[32];
};

struct arr2 {
  int x[32];
};

void f() {
  std::memory_order ord = std::memory_order_relaxed;

  int expected = 0, desired = 0;
  std::atomic<int> i{};
  i.operator=(0);
  i.store(0, ord);
  i.load(ord);
  i.operator int();
  i.exchange(0, ord);
  i.compare_exchange_weak(expected, desired, ord);
  i.compare_exchange_weak(expected, desired, ord, ord);
  i.compare_exchange_strong(expected, desired, ord);
  i.compare_exchange_strong(expected, desired, ord, ord);

  volatile std::atomic<int> vi{};
  vi.operator=(0);
  vi.store(0, ord);
  vi.load(ord);
  vi.operator int();
  vi.exchange(0, ord);
  vi.compare_exchange_weak(expected, desired, ord);
  vi.compare_exchange_weak(expected, desired, ord, ord);
  vi.compare_exchange_strong(expected, desired, ord);
  vi.compare_exchange_strong(expected, desired, ord, ord);

  arr test_value;

  volatile std::atomic<arr> va{};

  // expected-warning@*:* {{'__deprecated_if_not_always_lock_free<arr, false>' is deprecated: volatile atomic operations are deprecated when std::atomic<T>::is_always_lock_free is false}}
  va.operator=(test_value);

  // expected-warning@*:* {{'__deprecated_if_not_always_lock_free<arr, false>' is deprecated: volatile atomic operations are deprecated when std::atomic<T>::is_always_lock_free is false}}
  va.store(test_value, ord);

  // expected-warning@*:* {{'__deprecated_if_not_always_lock_free<arr, false>' is deprecated: volatile atomic operations are deprecated when std::atomic<T>::is_always_lock_free is false}}
  va.load(ord);

  // expected-warning@*:* {{'__deprecated_if_not_always_lock_free<arr, false>' is deprecated: volatile atomic operations are deprecated when std::atomic<T>::is_always_lock_free is false}}
  va.operator arr();

  // expected-warning@*:* {{'__deprecated_if_not_always_lock_free<arr, false>' is deprecated: volatile atomic operations are deprecated when std::atomic<T>::is_always_lock_free is false}}
  va.exchange(test_value, ord);

  // expected-warning@*:* {{'__deprecated_if_not_always_lock_free<arr, false>' is deprecated: volatile atomic operations are deprecated when std::atomic<T>::is_always_lock_free is false}}
  va.compare_exchange_weak(test_value, test_value, ord);

  // expected-warning@*:* {{'__deprecated_if_not_always_lock_free<arr, false>' is deprecated: volatile atomic operations are deprecated when std::atomic<T>::is_always_lock_free is false}}
  va.compare_exchange_weak(test_value, test_value, ord, ord);

  // expected-warning@*:* {{'__deprecated_if_not_always_lock_free<arr, false>' is deprecated: volatile atomic operations are deprecated when std::atomic<T>::is_always_lock_free is false}}
  va.compare_exchange_strong(test_value, test_value, ord);

  // expected-warning@*:* {{'__deprecated_if_not_always_lock_free<arr, false>' is deprecated: volatile atomic operations are deprecated when std::atomic<T>::is_always_lock_free is false}}
  va.compare_exchange_strong(test_value, test_value, ord, ord);

  const volatile std::atomic<arr2> cva{};

  // expected-warning@*:* {{'__deprecated_if_not_always_lock_free<arr2, false>' is deprecated: volatile atomic operations are deprecated when std::atomic<T>::is_always_lock_free is false}}
  cva.load(ord);

  // expected-warning@*:* {{'__deprecated_if_not_always_lock_free<arr2, false>' is deprecated: volatile atomic operations are deprecated when std::atomic<T>::is_always_lock_free is false}}
  cva.operator arr2();
}
