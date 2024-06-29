//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// XFAIL: availability-synchronization_library-missing

// <atomic>

// Tests the basic features and makes sure they work with a hijacking operator&.

//  template<class T> struct atomic {
//    using value_type = T;
//
//    static constexpr bool is_always_lock_free = implementation-defined;
//    bool is_lock_free() const volatile noexcept;
//    bool is_lock_free() const noexcept;
//
//    // [atomics.types.operations], operations on atomic types
//    constexpr atomic() noexcept(is_nothrow_default_constructible_v<T>);
//    constexpr atomic(T) noexcept;
//    atomic(const atomic&) = delete;
//    atomic& operator=(const atomic&) = delete;
//    atomic& operator=(const atomic&) volatile = delete;
//
//    T load(memory_order = memory_order::seq_cst) const volatile noexcept;
//    T load(memory_order = memory_order::seq_cst) const noexcept;
//    operator T() const volatile noexcept;
//    operator T() const noexcept;
//    void store(T, memory_order = memory_order::seq_cst) volatile noexcept;
//    void store(T, memory_order = memory_order::seq_cst) noexcept;
//    T operator=(T) volatile noexcept;
//    T operator=(T) noexcept;
//
//    T exchange(T, memory_order = memory_order::seq_cst) volatile noexcept;
//    T exchange(T, memory_order = memory_order::seq_cst) noexcept;
//    bool compare_exchange_weak(T&, T, memory_order, memory_order) volatile noexcept;
//    bool compare_exchange_weak(T&, T, memory_order, memory_order) noexcept;
//    bool compare_exchange_strong(T&, T, memory_order, memory_order) volatile noexcept;
//    bool compare_exchange_strong(T&, T, memory_order, memory_order) noexcept;
//    bool compare_exchange_weak(T&, T, memory_order = memory_order::seq_cst) volatile noexcept;
//    bool compare_exchange_weak(T&, T, memory_order = memory_order::seq_cst) noexcept;
//    bool compare_exchange_strong(T&, T, memory_order = memory_order::seq_cst) volatile noexcept;
//    bool compare_exchange_strong(T&, T, memory_order = memory_order::seq_cst) noexcept;
//
//    void wait(T, memory_order = memory_order::seq_cst) const volatile noexcept;
//    void wait(T, memory_order = memory_order::seq_cst) const noexcept;
//    void notify_one() volatile noexcept;
//    void notify_one() noexcept;
//    void notify_all() volatile noexcept;
//    void notify_all() noexcept;
//  };

#include <atomic>
#include <type_traits>

#include "operator_hijacker.h"
#include "test_macros.h"

template <class T>
void test() {
  T a;
  typename T::value_type v;
#if TEST_STD_VER >= 20
  std::memory_order m = std::memory_order::seq_cst;
#else
  std::memory_order m = std::memory_order_seq_cst;
#endif

  TEST_IGNORE_NODISCARD a.is_lock_free();

  TEST_DIAGNOSTIC_PUSH
  // MSVC warning C4197: 'volatile std::atomic<operator_hijacker>': top-level volatile in cast is ignored
  TEST_MSVC_DIAGNOSTIC_IGNORED(4197)

  TEST_IGNORE_NODISCARD T();
  TEST_IGNORE_NODISCARD T(v);

  TEST_DIAGNOSTIC_POP

  TEST_IGNORE_NODISCARD a.load();
  TEST_IGNORE_NODISCARD static_cast<typename T::value_type>(a);
  a.store(v);
  a = v;

  TEST_IGNORE_NODISCARD a.exchange(v);
  TEST_IGNORE_NODISCARD a.compare_exchange_weak(v, v, m, m);
  TEST_IGNORE_NODISCARD a.compare_exchange_strong(v, v, m, m);
  TEST_IGNORE_NODISCARD a.compare_exchange_weak(v, v);
  TEST_IGNORE_NODISCARD a.compare_exchange_strong(v, v, m);

  a.wait(v);
  a.notify_one();
  a.notify_all();
}

void test() {
  test<std::atomic<operator_hijacker>>();
  test<volatile std::atomic<operator_hijacker>>();
}
