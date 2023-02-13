//===-- trec_interface_atomic.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Public interface header for TSan atomics.
//===----------------------------------------------------------------------===//
#ifndef TREC_INTERFACE_ATOMIC_H
#define TREC_INTERFACE_ATOMIC_H

#ifdef __cplusplus
extern "C" {
#endif

typedef char __trec_atomic8;
typedef short __trec_atomic16;
typedef int __trec_atomic32;
typedef long __trec_atomic64;
#if defined(__SIZEOF_INT128__) \
    || (__clang_major__ * 100 + __clang_minor__ >= 302)
__extension__ typedef __int128 __trec_atomic128;
# define __TREC_HAS_INT128 1
#else
# define __TREC_HAS_INT128 0
#endif

// Part of ABI, do not change.
// https://github.com/llvm/llvm-project/blob/master/libcxx/include/atomic
typedef enum {
  __trec_memory_order_relaxed,
  __trec_memory_order_consume,
  __trec_memory_order_acquire,
  __trec_memory_order_release,
  __trec_memory_order_acq_rel,
  __trec_memory_order_seq_cst
} __trec_memory_order;

__trec_atomic8 __trec_atomic8_load(const volatile __trec_atomic8 *a,
    __trec_memory_order mo);
__trec_atomic16 __trec_atomic16_load(const volatile __trec_atomic16 *a,
    __trec_memory_order mo);
__trec_atomic32 __trec_atomic32_load(const volatile __trec_atomic32 *a,
    __trec_memory_order mo);
__trec_atomic64 __trec_atomic64_load(const volatile __trec_atomic64 *a,
    __trec_memory_order mo);
#if __TREC_HAS_INT128
__trec_atomic128 __trec_atomic128_load(const volatile __trec_atomic128 *a,
    __trec_memory_order mo);
#endif

void __trec_atomic8_store(volatile __trec_atomic8 *a, __trec_atomic8 v,
    __trec_memory_order mo);
void __trec_atomic16_store(volatile __trec_atomic16 *a, __trec_atomic16 v,
    __trec_memory_order mo);
void __trec_atomic32_store(volatile __trec_atomic32 *a, __trec_atomic32 v,
    __trec_memory_order mo);
void __trec_atomic64_store(volatile __trec_atomic64 *a, __trec_atomic64 v,
    __trec_memory_order mo);
#if __TREC_HAS_INT128
void __trec_atomic128_store(volatile __trec_atomic128 *a, __trec_atomic128 v,
    __trec_memory_order mo);
#endif

__trec_atomic8 __trec_atomic8_exchange(volatile __trec_atomic8 *a,
    __trec_atomic8 v, __trec_memory_order mo);
__trec_atomic16 __trec_atomic16_exchange(volatile __trec_atomic16 *a,
    __trec_atomic16 v, __trec_memory_order mo);
__trec_atomic32 __trec_atomic32_exchange(volatile __trec_atomic32 *a,
    __trec_atomic32 v, __trec_memory_order mo);
__trec_atomic64 __trec_atomic64_exchange(volatile __trec_atomic64 *a,
    __trec_atomic64 v, __trec_memory_order mo);
#if __TREC_HAS_INT128
__trec_atomic128 __trec_atomic128_exchange(volatile __trec_atomic128 *a,
    __trec_atomic128 v, __trec_memory_order mo);
#endif

__trec_atomic8 __trec_atomic8_fetch_add(volatile __trec_atomic8 *a,
    __trec_atomic8 v, __trec_memory_order mo);
__trec_atomic16 __trec_atomic16_fetch_add(volatile __trec_atomic16 *a,
    __trec_atomic16 v, __trec_memory_order mo);
__trec_atomic32 __trec_atomic32_fetch_add(volatile __trec_atomic32 *a,
    __trec_atomic32 v, __trec_memory_order mo);
__trec_atomic64 __trec_atomic64_fetch_add(volatile __trec_atomic64 *a,
    __trec_atomic64 v, __trec_memory_order mo);
#if __TREC_HAS_INT128
__trec_atomic128 __trec_atomic128_fetch_add(volatile __trec_atomic128 *a,
    __trec_atomic128 v, __trec_memory_order mo);
#endif

__trec_atomic8 __trec_atomic8_fetch_sub(volatile __trec_atomic8 *a,
    __trec_atomic8 v, __trec_memory_order mo);
__trec_atomic16 __trec_atomic16_fetch_sub(volatile __trec_atomic16 *a,
    __trec_atomic16 v, __trec_memory_order mo);
__trec_atomic32 __trec_atomic32_fetch_sub(volatile __trec_atomic32 *a,
    __trec_atomic32 v, __trec_memory_order mo);
__trec_atomic64 __trec_atomic64_fetch_sub(volatile __trec_atomic64 *a,
    __trec_atomic64 v, __trec_memory_order mo);
#if __TREC_HAS_INT128
__trec_atomic128 __trec_atomic128_fetch_sub(volatile __trec_atomic128 *a,
    __trec_atomic128 v, __trec_memory_order mo);
#endif

__trec_atomic8 __trec_atomic8_fetch_and(volatile __trec_atomic8 *a,
    __trec_atomic8 v, __trec_memory_order mo);
__trec_atomic16 __trec_atomic16_fetch_and(volatile __trec_atomic16 *a,
    __trec_atomic16 v, __trec_memory_order mo);
__trec_atomic32 __trec_atomic32_fetch_and(volatile __trec_atomic32 *a,
    __trec_atomic32 v, __trec_memory_order mo);
__trec_atomic64 __trec_atomic64_fetch_and(volatile __trec_atomic64 *a,
    __trec_atomic64 v, __trec_memory_order mo);
#if __TREC_HAS_INT128
__trec_atomic128 __trec_atomic128_fetch_and(volatile __trec_atomic128 *a,
    __trec_atomic128 v, __trec_memory_order mo);
#endif

__trec_atomic8 __trec_atomic8_fetch_or(volatile __trec_atomic8 *a,
    __trec_atomic8 v, __trec_memory_order mo);
__trec_atomic16 __trec_atomic16_fetch_or(volatile __trec_atomic16 *a,
    __trec_atomic16 v, __trec_memory_order mo);
__trec_atomic32 __trec_atomic32_fetch_or(volatile __trec_atomic32 *a,
    __trec_atomic32 v, __trec_memory_order mo);
__trec_atomic64 __trec_atomic64_fetch_or(volatile __trec_atomic64 *a,
    __trec_atomic64 v, __trec_memory_order mo);
#if __TREC_HAS_INT128
__trec_atomic128 __trec_atomic128_fetch_or(volatile __trec_atomic128 *a,
    __trec_atomic128 v, __trec_memory_order mo);
#endif

__trec_atomic8 __trec_atomic8_fetch_xor(volatile __trec_atomic8 *a,
    __trec_atomic8 v, __trec_memory_order mo);
__trec_atomic16 __trec_atomic16_fetch_xor(volatile __trec_atomic16 *a,
    __trec_atomic16 v, __trec_memory_order mo);
__trec_atomic32 __trec_atomic32_fetch_xor(volatile __trec_atomic32 *a,
    __trec_atomic32 v, __trec_memory_order mo);
__trec_atomic64 __trec_atomic64_fetch_xor(volatile __trec_atomic64 *a,
    __trec_atomic64 v, __trec_memory_order mo);
#if __TREC_HAS_INT128
__trec_atomic128 __trec_atomic128_fetch_xor(volatile __trec_atomic128 *a,
    __trec_atomic128 v, __trec_memory_order mo);
#endif

__trec_atomic8 __trec_atomic8_fetch_nand(volatile __trec_atomic8 *a,
    __trec_atomic8 v, __trec_memory_order mo);
__trec_atomic16 __trec_atomic16_fetch_nand(volatile __trec_atomic16 *a,
    __trec_atomic16 v, __trec_memory_order mo);
__trec_atomic32 __trec_atomic32_fetch_nand(volatile __trec_atomic32 *a,
    __trec_atomic32 v, __trec_memory_order mo);
__trec_atomic64 __trec_atomic64_fetch_nand(volatile __trec_atomic64 *a,
    __trec_atomic64 v, __trec_memory_order mo);
#if __TREC_HAS_INT128
__trec_atomic128 __trec_atomic128_fetch_nand(volatile __trec_atomic128 *a,
    __trec_atomic128 v, __trec_memory_order mo);
#endif

int __trec_atomic8_compare_exchange_weak(volatile __trec_atomic8 *a,
    __trec_atomic8 *c, __trec_atomic8 v, __trec_memory_order mo,
    __trec_memory_order fail_mo);
int __trec_atomic16_compare_exchange_weak(volatile __trec_atomic16 *a,
    __trec_atomic16 *c, __trec_atomic16 v, __trec_memory_order mo,
    __trec_memory_order fail_mo);
int __trec_atomic32_compare_exchange_weak(volatile __trec_atomic32 *a,
    __trec_atomic32 *c, __trec_atomic32 v, __trec_memory_order mo,
    __trec_memory_order fail_mo);
int __trec_atomic64_compare_exchange_weak(volatile __trec_atomic64 *a,
    __trec_atomic64 *c, __trec_atomic64 v, __trec_memory_order mo,
    __trec_memory_order fail_mo);
#if __TREC_HAS_INT128
int __trec_atomic128_compare_exchange_weak(volatile __trec_atomic128 *a,
    __trec_atomic128 *c, __trec_atomic128 v, __trec_memory_order mo,
    __trec_memory_order fail_mo);
#endif

int __trec_atomic8_compare_exchange_strong(volatile __trec_atomic8 *a,
    __trec_atomic8 *c, __trec_atomic8 v, __trec_memory_order mo,
    __trec_memory_order fail_mo);
int __trec_atomic16_compare_exchange_strong(volatile __trec_atomic16 *a,
    __trec_atomic16 *c, __trec_atomic16 v, __trec_memory_order mo,
    __trec_memory_order fail_mo);
int __trec_atomic32_compare_exchange_strong(volatile __trec_atomic32 *a,
    __trec_atomic32 *c, __trec_atomic32 v, __trec_memory_order mo,
    __trec_memory_order fail_mo);
int __trec_atomic64_compare_exchange_strong(volatile __trec_atomic64 *a,
    __trec_atomic64 *c, __trec_atomic64 v, __trec_memory_order mo,
    __trec_memory_order fail_mo);
#if __TREC_HAS_INT128
int __trec_atomic128_compare_exchange_strong(volatile __trec_atomic128 *a,
    __trec_atomic128 *c, __trec_atomic128 v, __trec_memory_order mo,
    __trec_memory_order fail_mo);
#endif

__trec_atomic8 __trec_atomic8_compare_exchange_val(
    volatile __trec_atomic8 *a, __trec_atomic8 c, __trec_atomic8 v,
    __trec_memory_order mo, __trec_memory_order fail_mo);
__trec_atomic16 __trec_atomic16_compare_exchange_val(
    volatile __trec_atomic16 *a, __trec_atomic16 c, __trec_atomic16 v,
    __trec_memory_order mo, __trec_memory_order fail_mo);
__trec_atomic32 __trec_atomic32_compare_exchange_val(
    volatile __trec_atomic32 *a, __trec_atomic32 c, __trec_atomic32 v,
    __trec_memory_order mo, __trec_memory_order fail_mo);
__trec_atomic64 __trec_atomic64_compare_exchange_val(
    volatile __trec_atomic64 *a, __trec_atomic64 c, __trec_atomic64 v,
    __trec_memory_order mo, __trec_memory_order fail_mo);
#if __TREC_HAS_INT128
__trec_atomic128 __trec_atomic128_compare_exchange_val(
    volatile __trec_atomic128 *a, __trec_atomic128 c, __trec_atomic128 v,
    __trec_memory_order mo, __trec_memory_order fail_mo);
#endif

void __trec_atomic_thread_fence(__trec_memory_order mo);
void __trec_atomic_signal_fence(__trec_memory_order mo);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TREC_INTERFACE_ATOMIC_H
