//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support
// UNSUPPORTED: libcpp-has-no-threads

// <atomic>

// Test that null pointer parameters are diagnosed.

#include <atomic>

int main() {
  std::atomic<int> ai;
  volatile std::atomic<int> vai;
  int i = 42;

  atomic_is_lock_free((const volatile std::atomic<int>*)0); // expected-error {{null passed to a callee that requires a non-null argument}}
  atomic_is_lock_free((const std::atomic<int>*)0); // expected-error {{null passed to a callee that requires a non-null argument}}
  atomic_init((volatile std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  atomic_init((std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  atomic_store((volatile std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  atomic_store((std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  atomic_store_explicit((volatile std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  atomic_store_explicit((std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_load((const volatile std::atomic<int>*)0); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_load((const std::atomic<int>*)0); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_load_explicit((const volatile std::atomic<int>*)0, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_load_explicit((const std::atomic<int>*)0, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_exchange((volatile std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_exchange((std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_exchange_explicit((volatile std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_exchange_explicit((std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_weak((volatile std::atomic<int>*)0, &i, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_weak((std::atomic<int>*)0, &i, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_strong((volatile std::atomic<int>*)0, &i, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_strong((std::atomic<int>*)0, &i, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_weak(&vai, (int*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_weak(&ai, (int*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_strong(&vai, (int*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_strong(&ai, (int*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_weak_explicit((volatile std::atomic<int>*)0, &i, 42, std::memory_order_relaxed, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_weak_explicit((std::atomic<int>*)0, &i, 42, std::memory_order_relaxed, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_strong_explicit((volatile std::atomic<int>*)0, &i, 42, std::memory_order_relaxed, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_strong_explicit((std::atomic<int>*)0, &i, 42, std::memory_order_relaxed, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_weak_explicit(&vai, (int*)0, 42, std::memory_order_relaxed, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_weak_explicit(&ai, (int*)0, 42, std::memory_order_relaxed, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_strong_explicit(&vai, (int*)0, 42, std::memory_order_relaxed, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_compare_exchange_strong_explicit(&ai, (int*)0, 42, std::memory_order_relaxed, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_add((volatile std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_add((std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_add((volatile std::atomic<int*>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_add((std::atomic<int*>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_add_explicit((volatile std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_add_explicit((std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_add_explicit((volatile std::atomic<int*>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_add_explicit((std::atomic<int*>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_sub((volatile std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_sub((std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_sub((volatile std::atomic<int*>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_sub((std::atomic<int*>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_sub_explicit((volatile std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_sub_explicit((std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_sub_explicit((volatile std::atomic<int*>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_sub_explicit((std::atomic<int*>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_and((volatile std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_and((std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_and_explicit((volatile std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_and_explicit((std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_or((volatile std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_or((std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_or_explicit((volatile std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_or_explicit((std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_xor((volatile std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_xor((std::atomic<int>*)0, 42); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_xor_explicit((volatile std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_fetch_xor_explicit((std::atomic<int>*)0, 42, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_flag_test_and_set((volatile std::atomic_flag*)0); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_flag_test_and_set((std::atomic_flag*)0); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_flag_test_and_set_explicit((volatile std::atomic_flag*)0, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_flag_test_and_set_explicit((std::atomic_flag*)0, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_flag_clear((volatile std::atomic_flag*)0); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_flag_clear((std::atomic_flag*)0); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_flag_clear_explicit((volatile std::atomic_flag*)0, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
  (void)atomic_flag_clear_explicit((std::atomic_flag*)0, std::memory_order_relaxed); // expected-error {{null passed to a callee that requires a non-null argument}}
}
