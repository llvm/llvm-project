//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-threads

// <stdatomic.h>

// template<class T>
//   using std-atomic = std::atomic<T>;        // exposition only
//
// #define _Atomic(T) std-atomic<T>
//
// #define ATOMIC_BOOL_LOCK_FREE see below
// #define ATOMIC_CHAR_LOCK_FREE see below
// #define ATOMIC_CHAR16_T_LOCK_FREE see below
// #define ATOMIC_CHAR32_T_LOCK_FREE see below
// #define ATOMIC_WCHAR_T_LOCK_FREE see below
// #define ATOMIC_SHORT_LOCK_FREE see below
// #define ATOMIC_INT_LOCK_FREE see below
// #define ATOMIC_LONG_LOCK_FREE see below
// #define ATOMIC_LLONG_LOCK_FREE see below
// #define ATOMIC_POINTER_LOCK_FREE see below
//
// using std::memory_order                // see below
// using std::memory_order_relaxed        // see below
// using std::memory_order_consume        // see below
// using std::memory_order_acquire        // see below
// using std::memory_order_release        // see below
// using std::memory_order_acq_rel        // see below
// using std::memory_order_seq_cst        // see below
//
// using std::atomic_flag                 // see below
//
// using std::atomic_bool                 // see below
// using std::atomic_char                 // see below
// using std::atomic_schar                // see below
// using std::atomic_uchar                // see below
// using std::atomic_short                // see below
// using std::atomic_ushort               // see below
// using std::atomic_int                  // see below
// using std::atomic_uint                 // see below
// using std::atomic_long                 // see below
// using std::atomic_ulong                // see below
// using std::atomic_llong                // see below
// using std::atomic_ullong               // see below
// using std::atomic_char8_t              // see below
// using std::atomic_char16_t             // see below
// using std::atomic_char32_t             // see below
// using std::atomic_wchar_t              // see below
// using std::atomic_int8_t               // see below
// using std::atomic_uint8_t              // see below
// using std::atomic_int16_t              // see below
// using std::atomic_uint16_t             // see below
// using std::atomic_int32_t              // see below
// using std::atomic_uint32_t             // see below
// using std::atomic_int64_t              // see below
// using std::atomic_uint64_t             // see below
// using std::atomic_int_least8_t         // see below
// using std::atomic_uint_least8_t        // see below
// using std::atomic_int_least16_t        // see below
// using std::atomic_uint_least16_t       // see below
// using std::atomic_int_least32_t        // see below
// using std::atomic_uint_least32_t       // see below
// using std::atomic_int_least64_t        // see below
// using std::atomic_uint_least64_t       // see below
// using std::atomic_int_fast8_t          // see below
// using std::atomic_uint_fast8_t         // see below
// using std::atomic_int_fast16_t         // see below
// using std::atomic_uint_fast16_t        // see below
// using std::atomic_int_fast32_t         // see below
// using std::atomic_uint_fast32_t        // see below
// using std::atomic_int_fast64_t         // see below
// using std::atomic_uint_fast64_t        // see below
// using std::atomic_intptr_t             // see below
// using std::atomic_uintptr_t            // see below
// using std::atomic_size_t               // see below
// using std::atomic_ptrdiff_t            // see below
// using std::atomic_intmax_t             // see below
// using std::atomic_uintmax_t            // see below
//
// using std::atomic_is_lock_free                         // see below
// using std::atomic_load                                 // see below
// using std::atomic_load_explicit                        // see below
// using std::atomic_store                                // see below
// using std::atomic_store_explicit                       // see below
// using std::atomic_exchange                             // see below
// using std::atomic_exchange_explicit                    // see below
// using std::atomic_compare_exchange_strong              // see below
// using std::atomic_compare_exchange_strong_explicit     // see below
// using std::atomic_compare_exchange_weak                // see below
// using std::atomic_compare_exchange_weak_explicit       // see below
// using std::atomic_fetch_add                            // see below
// using std::atomic_fetch_add_explicit                   // see below
// using std::atomic_fetch_sub                            // see below
// using std::atomic_fetch_sub_explicit                   // see below
// using std::atomic_fetch_or                             // see below
// using std::atomic_fetch_or_explicit                    // see below
// using std::atomic_fetch_and                            // see below
// using std::atomic_fetch_and_explicit                   // see below
// using std::atomic_flag_test_and_set                    // see below
// using std::atomic_flag_test_and_set_explicit           // see below
// using std::atomic_flag_clear                           // see below
// using std::atomic_flag_clear_explicit                  // see below
//
// using std::atomic_thread_fence                         // see below
// using std::atomic_signal_fence                         // see below

#include <stdatomic.h>
#include <type_traits>

#include "test_macros.h"

static_assert(std::atomic<bool>::is_always_lock_free == (2 == ATOMIC_BOOL_LOCK_FREE));
static_assert(std::atomic<char>::is_always_lock_free == (2 == ATOMIC_CHAR_LOCK_FREE));
static_assert(std::atomic<signed char>::is_always_lock_free == (2 == ATOMIC_CHAR_LOCK_FREE));
static_assert(std::atomic<unsigned char>::is_always_lock_free == (2 == ATOMIC_CHAR_LOCK_FREE));
static_assert(std::atomic<char16_t>::is_always_lock_free == (2 == ATOMIC_CHAR16_T_LOCK_FREE));
static_assert(std::atomic<char32_t>::is_always_lock_free == (2 == ATOMIC_CHAR32_T_LOCK_FREE));
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(std::atomic<wchar_t>::is_always_lock_free == (2 == ATOMIC_WCHAR_T_LOCK_FREE));
#endif
static_assert(std::atomic<short>::is_always_lock_free == (2 == ATOMIC_SHORT_LOCK_FREE));
static_assert(std::atomic<unsigned short>::is_always_lock_free == (2 == ATOMIC_SHORT_LOCK_FREE));
static_assert(std::atomic<int>::is_always_lock_free == (2 == ATOMIC_INT_LOCK_FREE));
static_assert(std::atomic<unsigned int>::is_always_lock_free == (2 == ATOMIC_INT_LOCK_FREE));
static_assert(std::atomic<long>::is_always_lock_free == (2 == ATOMIC_LONG_LOCK_FREE));
static_assert(std::atomic<unsigned long>::is_always_lock_free == (2 == ATOMIC_LONG_LOCK_FREE));
static_assert(std::atomic<long long>::is_always_lock_free == (2 == ATOMIC_LLONG_LOCK_FREE));
static_assert(std::atomic<unsigned long long>::is_always_lock_free == (2 == ATOMIC_LLONG_LOCK_FREE));
static_assert(std::atomic<void*>::is_always_lock_free == (2 == ATOMIC_POINTER_LOCK_FREE));
static_assert(std::atomic<std::nullptr_t>::is_always_lock_free == (2 == ATOMIC_POINTER_LOCK_FREE));

void f() {
  static_assert(std::is_same_v<std::atomic<char>, _Atomic(char)>);
  static_assert(std::is_same_v<std::atomic<int>, _Atomic(int)>);
  static_assert(std::is_same_v<std::atomic<const long>, _Atomic(const long)>);

  static_assert(std::is_same_v<std::memory_order, ::memory_order>);
  static_assert(std::memory_order_relaxed == ::memory_order_relaxed);
  static_assert(std::memory_order_consume == ::memory_order_consume);
  static_assert(std::memory_order_acquire == ::memory_order_acquire);
  static_assert(std::memory_order_release == ::memory_order_release);
  static_assert(std::memory_order_acq_rel == ::memory_order_acq_rel);
  static_assert(std::memory_order_seq_cst == ::memory_order_seq_cst);

  static_assert(std::is_same_v<std::atomic_flag, ::atomic_flag>);

  static_assert(std::is_same_v<std::atomic<bool>, ::atomic_bool>);
  static_assert(std::is_same_v<std::atomic<char>, ::atomic_char>);
  static_assert(std::is_same_v<std::atomic<signed char>, ::atomic_schar>);
  static_assert(std::is_same_v<std::atomic<unsigned char>, ::atomic_uchar>);
  static_assert(std::is_same_v<std::atomic<short>, ::atomic_short>);
  static_assert(std::is_same_v<std::atomic<unsigned short>, ::atomic_ushort>);
  static_assert(std::is_same_v<std::atomic<int>, ::atomic_int>);
  static_assert(std::is_same_v<std::atomic<unsigned int>, ::atomic_uint>);
  static_assert(std::is_same_v<std::atomic<long>, ::atomic_long>);
  static_assert(std::is_same_v<std::atomic<unsigned long>, ::atomic_ulong>);
  static_assert(std::is_same_v<std::atomic<long long>, ::atomic_llong>);
  static_assert(std::is_same_v<std::atomic<unsigned long long>, ::atomic_ullong>);

#ifndef _LIBCPP_HAS_NO_CHAR8_T
  static_assert(std::is_same_v<std::atomic<char8_t>,  ::atomic_char8_t>);
#endif
  static_assert(std::is_same_v<std::atomic<char16_t>, ::atomic_char16_t>);
  static_assert(std::is_same_v<std::atomic<char32_t>, ::atomic_char32_t>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  static_assert(std::is_same_v<std::atomic<wchar_t>,  ::atomic_wchar_t>);
#endif

  static_assert(std::is_same_v<std::atomic<int8_t>,   ::atomic_int8_t>);
  static_assert(std::is_same_v<std::atomic<uint8_t>,  ::atomic_uint8_t>);
  static_assert(std::is_same_v<std::atomic<int16_t>,  ::atomic_int16_t>);
  static_assert(std::is_same_v<std::atomic<uint16_t>, ::atomic_uint16_t>);
  static_assert(std::is_same_v<std::atomic<int32_t>,  ::atomic_int32_t>);
  static_assert(std::is_same_v<std::atomic<uint32_t>, ::atomic_uint32_t>);
  static_assert(std::is_same_v<std::atomic<int64_t>,  ::atomic_int64_t>);
  static_assert(std::is_same_v<std::atomic<uint64_t>, ::atomic_uint64_t>);

  static_assert(std::is_same_v<std::atomic<int_least8_t>,   ::atomic_int_least8_t>);
  static_assert(std::is_same_v<std::atomic<uint_least8_t>,  ::atomic_uint_least8_t>);
  static_assert(std::is_same_v<std::atomic<int_least16_t>,  ::atomic_int_least16_t>);
  static_assert(std::is_same_v<std::atomic<uint_least16_t>, ::atomic_uint_least16_t>);
  static_assert(std::is_same_v<std::atomic<int_least32_t>,  ::atomic_int_least32_t>);
  static_assert(std::is_same_v<std::atomic<uint_least32_t>, ::atomic_uint_least32_t>);
  static_assert(std::is_same_v<std::atomic<int_least64_t>,  ::atomic_int_least64_t>);
  static_assert(std::is_same_v<std::atomic<uint_least64_t>, ::atomic_uint_least64_t>);

  static_assert(std::is_same_v<std::atomic<int_fast8_t>,    ::atomic_int_fast8_t>);
  static_assert(std::is_same_v<std::atomic<uint_fast8_t>,   ::atomic_uint_fast8_t>);
  static_assert(std::is_same_v<std::atomic<int_fast16_t>,   ::atomic_int_fast16_t>);
  static_assert(std::is_same_v<std::atomic<uint_fast16_t>,  ::atomic_uint_fast16_t>);
  static_assert(std::is_same_v<std::atomic<int_fast32_t>,   ::atomic_int_fast32_t>);
  static_assert(std::is_same_v<std::atomic<uint_fast32_t>,  ::atomic_uint_fast32_t>);
  static_assert(std::is_same_v<std::atomic<int_fast64_t>,   ::atomic_int_fast64_t>);
  static_assert(std::is_same_v<std::atomic<uint_fast64_t>,  ::atomic_uint_fast64_t>);

  static_assert(std::is_same_v<std::atomic<intptr_t>,  ::atomic_intptr_t>);
  static_assert(std::is_same_v<std::atomic<uintptr_t>, ::atomic_uintptr_t>);
  static_assert(std::is_same_v<std::atomic<size_t>,    ::atomic_size_t>);
  static_assert(std::is_same_v<std::atomic<ptrdiff_t>, ::atomic_ptrdiff_t>);
  static_assert(std::is_same_v<std::atomic<intmax_t>,  ::atomic_intmax_t>);
  static_assert(std::is_same_v<std::atomic<uintmax_t>, ::atomic_uintmax_t>);

  // Just check that the symbols in the global namespace are visible.
  using ::atomic_compare_exchange_strong;
  using ::atomic_compare_exchange_strong_explicit;
  using ::atomic_compare_exchange_weak;
  using ::atomic_compare_exchange_weak_explicit;
  using ::atomic_exchange;
  using ::atomic_exchange_explicit;
  using ::atomic_fetch_add;
  using ::atomic_fetch_add_explicit;
  using ::atomic_fetch_and;
  using ::atomic_fetch_and_explicit;
  using ::atomic_fetch_or;
  using ::atomic_fetch_or_explicit;
  using ::atomic_fetch_sub;
  using ::atomic_fetch_sub_explicit;
  using ::atomic_flag_clear;
  using ::atomic_flag_clear_explicit;
  using ::atomic_flag_test_and_set;
  using ::atomic_flag_test_and_set_explicit;
  using ::atomic_is_lock_free;
  using ::atomic_load;
  using ::atomic_load_explicit;
  using ::atomic_store;
  using ::atomic_store_explicit;

  using ::atomic_signal_fence;
  using ::atomic_thread_fence;
}
