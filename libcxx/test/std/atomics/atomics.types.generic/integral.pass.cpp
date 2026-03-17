//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: !has-64-bit-atomics

// <atomic>

// template <>
// struct atomic<integral>
// {
//     bool is_lock_free() const volatile;
//     bool is_lock_free() const;
//     void store(integral desr, memory_order m = memory_order_seq_cst) volatile;
//     void store(integral desr, memory_order m = memory_order_seq_cst);
//     integral load(memory_order m = memory_order_seq_cst) const volatile;
//     integral load(memory_order m = memory_order_seq_cst) const;
//     operator integral() const volatile;
//     operator integral() const;
//     integral exchange(integral desr,
//                       memory_order m = memory_order_seq_cst) volatile;
//     integral exchange(integral desr, memory_order m = memory_order_seq_cst);
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order s, memory_order f) volatile;
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order s, memory_order f);
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                  memory_order s, memory_order f) volatile;
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                  memory_order s, memory_order f);
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_weak(integral& expc, integral desr,
//                                memory_order m = memory_order_seq_cst);
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                 memory_order m = memory_order_seq_cst) volatile;
//     bool compare_exchange_strong(integral& expc, integral desr,
//                                  memory_order m = memory_order_seq_cst);
//
//     integral
//         fetch_add(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_add(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_sub(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_sub(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_and(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_and(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_or(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_or(integral op, memory_order m = memory_order_seq_cst);
//     integral
//         fetch_xor(integral op, memory_order m = memory_order_seq_cst) volatile;
//     integral fetch_xor(integral op, memory_order m = memory_order_seq_cst);
//
//     void
//         store_add(integral op, memory_order m = memory_order_seq_cst) volatile;
//     void store_add(integral op, memory_order m = memory_order_seq_cst);
//     void
//         store_sub(integral op, memory_order m = memory_order_seq_cst) volatile;
//     void store_sub(integral op, memory_order m = memory_order_seq_cst);
//     void
//         store_and(integral op, memory_order m = memory_order_seq_cst) volatile;
//     void store_and(integral op, memory_order m = memory_order_seq_cst);
//     void
//         store_or(integral op, memory_order m = memory_order_seq_cst) volatile;
//     void store_or(integral op, memory_order m = memory_order_seq_cst);
//     void
//         store_xor(integral op, memory_order m = memory_order_seq_cst) volatile;
//     void store_xor(integral op, memory_order m = memory_order_seq_cst);
//
//     atomic() = default;
//     constexpr atomic(integral desr);
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
//     integral operator=(integral desr) volatile;
//     integral operator=(integral desr);
//
//     integral operator++(int) volatile;
//     integral operator++(int);
//     integral operator--(int) volatile;
//     integral operator--(int);
//     integral operator++() volatile;
//     integral operator++();
//     integral operator--() volatile;
//     integral operator--();
//     integral operator+=(integral op) volatile;
//     integral operator+=(integral op);
//     integral operator-=(integral op) volatile;
//     integral operator-=(integral op);
//     integral operator&=(integral op) volatile;
//     integral operator&=(integral op);
//     integral operator|=(integral op) volatile;
//     integral operator|=(integral op);
//     integral operator^=(integral op) volatile;
//     integral operator^=(integral op);
// };

#include <atomic>
#include <cassert>
#include <cstdint>
#include <new>

#include <cmpxchg_loop.h>

#include "test_macros.h"

template <class A, class T>
void
do_test()
{
    A obj(T(0));
    assert(obj == T(0));
    {
        bool lockfree = obj.is_lock_free();
        (void)lockfree;
#if TEST_STD_VER >= 17
        if (A::is_always_lock_free)
            assert(lockfree);
#endif
    }
    obj.store(T(0));
    assert(obj == T(0));
    obj.store(T(1), std::memory_order_release);
    assert(obj == T(1));
    assert(obj.load() == T(1));
    assert(obj.load(std::memory_order_acquire) == T(1));
    assert(obj.exchange(T(2)) == T(1));
    assert(obj == T(2));
    assert(obj.exchange(T(3), std::memory_order_relaxed) == T(2));
    assert(obj == T(3));
    T x = obj;
    assert(cmpxchg_weak_loop(obj, x, T(2)) == true);
    assert(obj == T(2));
    assert(x == T(3));
    assert(obj.compare_exchange_weak(x, T(1)) == false);
    assert(obj == T(2));
    assert(x == T(2));
    x = T(2);
    assert(obj.compare_exchange_strong(x, T(1)) == true);
    assert(obj == T(1));
    assert(x == T(2));
    assert(obj.compare_exchange_strong(x, T(0)) == false);
    assert(obj == T(1));
    assert(x == T(1));
    assert((obj = T(0)) == T(0));
    assert(obj == T(0));
    assert(obj++ == T(0));
    assert(obj == T(1));
    assert(++obj == T(2));
    assert(obj == T(2));
    assert(--obj == T(1));
    assert(obj == T(1));
    assert(obj-- == T(1));
    assert(obj == T(0));
    obj = T(2);
    assert((obj += T(3)) == T(5));
    assert(obj == T(5));
    assert((obj -= T(3)) == T(2));
    assert(obj == T(2));
    assert((obj |= T(5)) == T(7));
    assert(obj == T(7));
    assert((obj &= T(0xF)) == T(7));
    assert(obj == T(7));
    assert((obj ^= T(0xF)) == T(8));
    assert(obj == T(8));

#if TEST_STD_VER >= 26
    // Test store_add
    obj.store(T(1));
    obj.store_add(T(2));
    assert(obj == T(3));
    obj.store_add(T(0), std::memory_order_relaxed);
    assert(obj == T(3));

    // Test store_sub
    obj.store(T(10));
    obj.store_sub(T(3));
    assert(obj == T(7));
    obj.store_sub(T(0), std::memory_order_release);
    assert(obj == T(7));

    // Test store_and
    obj.store(T(0b1111));
    obj.store_and(T(0b1010));
    assert(obj == T(0b1010));
    obj.store_and(T(0b1111), std::memory_order_relaxed);
    assert(obj == T(0b1010));

    // Test store_or
    obj.store(T(0b1010));
    obj.store_or(T(0b0101));
    assert(obj == T(0b1111));
    obj.store_or(T(0b0000), std::memory_order_release);
    assert(obj == T(0b1111));

    // Test store_xor
    obj.store(T(0b1100));
    obj.store_xor(T(0b1010));
    assert(obj == T(0b0110));
    obj.store_xor(T(0b0000), std::memory_order_relaxed);
    assert(obj == T(0b0110));
#endif

    {
        TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {23};
        A& zero = *new (storage) A();
        assert(zero == 0);
        zero.~A();
    }
}

template <class A, class T>
void test()
{
    do_test<A, T>();
    do_test<volatile A, T>();
}


int main(int, char**)
{
    test<std::atomic_char, char>();
    test<std::atomic_schar, signed char>();
    test<std::atomic_uchar, unsigned char>();
    test<std::atomic_short, short>();
    test<std::atomic_ushort, unsigned short>();
    test<std::atomic_int, int>();
    test<std::atomic_uint, unsigned int>();
    test<std::atomic_long, long>();
    test<std::atomic_ulong, unsigned long>();
    test<std::atomic_llong, long long>();
    test<std::atomic_ullong, unsigned long long>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<std::atomic_char8_t, char8_t>();
#endif
    test<std::atomic_char16_t, char16_t>();
    test<std::atomic_char32_t, char32_t>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::atomic_wchar_t, wchar_t>();
#endif

    test<std::atomic_int8_t,    int8_t>();
    test<std::atomic_uint8_t,  uint8_t>();
    test<std::atomic_int16_t,   int16_t>();
    test<std::atomic_uint16_t, uint16_t>();
    test<std::atomic_int32_t,   int32_t>();
    test<std::atomic_uint32_t, uint32_t>();
    test<std::atomic_int64_t,   int64_t>();
    test<std::atomic_uint64_t, uint64_t>();

    test<volatile std::atomic_char, char>();
    test<volatile std::atomic_schar, signed char>();
    test<volatile std::atomic_uchar, unsigned char>();
    test<volatile std::atomic_short, short>();
    test<volatile std::atomic_ushort, unsigned short>();
    test<volatile std::atomic_int, int>();
    test<volatile std::atomic_uint, unsigned int>();
    test<volatile std::atomic_long, long>();
    test<volatile std::atomic_ulong, unsigned long>();
    test<volatile std::atomic_llong, long long>();
    test<volatile std::atomic_ullong, unsigned long long>();
    test<volatile std::atomic_char16_t, char16_t>();
    test<volatile std::atomic_char32_t, char32_t>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<volatile std::atomic_wchar_t, wchar_t>();
#endif

    test<volatile std::atomic_int8_t,    int8_t>();
    test<volatile std::atomic_uint8_t,  uint8_t>();
    test<volatile std::atomic_int16_t,   int16_t>();
    test<volatile std::atomic_uint16_t, uint16_t>();
    test<volatile std::atomic_int32_t,   int32_t>();
    test<volatile std::atomic_uint32_t, uint32_t>();
    test<volatile std::atomic_int64_t,   int64_t>();
    test<volatile std::atomic_uint64_t, uint64_t>();

  return 0;
}
