//===-- trec_interface.h ----------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
// The functions declared in this header will be inserted by the instrumentation
// module.
// This header can be included by the instrumented program or by TRec
// tests.
//===----------------------------------------------------------------------===//
#ifndef TREC_INTERFACE_H
#define TREC_INTERFACE_H

#include <sanitizer_common/sanitizer_internal_defs.h>
using __sanitizer::tid_t;
using __sanitizer::u32;
using __sanitizer::uptr;

// This header should NOT include any other headers.
// All functions in this header are extern "C" and start with __trec_.

#ifdef __cplusplus
extern "C" {
#endif

#if !SANITIZER_GO

// This function should be called at the very beginning of the process,
// before any instrumented code is executed and before any call to malloc.
SANITIZER_INTERFACE_ATTRIBUTE void __trec_init();

SANITIZER_INTERFACE_ATTRIBUTE void __trec_branch(__sanitizer::u64 cond);

SANITIZER_INTERFACE_ATTRIBUTE void __trec_func_param(__sanitizer::u16 param_idx,
                                                     void *src_addr,
                                                     __sanitizer::u16 src_idx,
                                                     void *val);

SANITIZER_INTERFACE_ATTRIBUTE void __trec_func_exit_param(
    void *src_addr, __sanitizer::u16 src_idx, void *val);

SANITIZER_INTERFACE_ATTRIBUTE void __trec_inst_debug_info(__sanitizer::u32 line,
                                                          __sanitizer::u16 col,
                                                          char *name1,
                                                          char *name2);

SANITIZER_INTERFACE_ATTRIBUTE void __trec_read1(void *addr, bool isPtr,
                                                void *val, void *addr_src_addr,
                                                __sanitizer::u16 addr_src_idx);
SANITIZER_INTERFACE_ATTRIBUTE void __trec_read2(void *addr, bool isPtr,
                                                void *val, void *addr_src_addr,
                                                __sanitizer::u16 addr_src_idx);
SANITIZER_INTERFACE_ATTRIBUTE void __trec_read4(void *addr, bool isPtr,
                                                void *val, void *addr_src_addr,
                                                __sanitizer::u16 addr_src_idx);
SANITIZER_INTERFACE_ATTRIBUTE void __trec_read8(void *addr, bool isPtr,
                                                void *val, void *addr_src_addr,
                                                __sanitizer::u16 addr_src_idx);
#if TREC_HAS_128_BIT
SANITIZER_INTERFACE_ATTRIBUTE void __trec_read16(void *addr, bool isPtr,
                                                 __uint128_t val,
                                                 void *addr_src_addr,
                                                 __sanitizer::u16 addr_src_idx);
#endif

SANITIZER_INTERFACE_ATTRIBUTE void __trec_write1(void *addr, bool isPtr,
                                                 void *val, void *addr_src_addr,
                                                 __sanitizer::u16 addr_src_idx,
                                                 void *val_src_addr,
                                                 __sanitizer::u16 val_src_idx);
SANITIZER_INTERFACE_ATTRIBUTE void __trec_write2(void *addr, bool isPtr,
                                                 void *val, void *addr_src_addr,
                                                 __sanitizer::u16 addr_src_idx,
                                                 void *val_src_addr,
                                                 __sanitizer::u16 val_src_idx);
SANITIZER_INTERFACE_ATTRIBUTE void __trec_write4(void *addr, bool isPtr,
                                                 void *val, void *addr_src_addr,
                                                 __sanitizer::u16 addr_src_idx,
                                                 void *val_src_addr,
                                                 __sanitizer::u16 val_src_idx);
SANITIZER_INTERFACE_ATTRIBUTE void __trec_write8(void *addr, bool isPtr,
                                                 void *val, void *addr_src_addr,
                                                 __sanitizer::u16 addr_src_idx,
                                                 void *val_src_addr,
                                                 __sanitizer::u16 val_src_idx);
#if TREC_HAS_128_BIT
SANITIZER_INTERFACE_ATTRIBUTE void __trec_write16(void *addr, bool isPtr,
                                                  __uint128_t val,
                                                  void *addr_src_addr,
                                                  __sanitizer::u16 addr_src_idx,
                                                  void *val_src_addr,
                                                  __sanitizer::u16 val_src_idx);
#endif
SANITIZER_INTERFACE_ATTRIBUTE void __trec_unaligned_read2(
    const void *addr, bool isPtr, void *val, void *addr_src_addr,
    __sanitizer::u16 addr_src_idx);
SANITIZER_INTERFACE_ATTRIBUTE void __trec_unaligned_read4(
    const void *addr, bool isPtr, void *val, void *addr_src_addr,
    __sanitizer::u16 addr_src_idx);
SANITIZER_INTERFACE_ATTRIBUTE void __trec_unaligned_read8(
    const void *addr, bool isPtr, void *val, void *addr_src_addr,
    __sanitizer::u16 addr_src_idx);
#if TREC_HAS_128_BIT
SANITIZER_INTERFACE_ATTRIBUTE void __trec_unaligned_read16(
    const void *addr, bool isPtr, __uint128_t val, void *addr_src_addr,
    __sanitizer::u16 addr_src_idx);
#endif
SANITIZER_INTERFACE_ATTRIBUTE void __trec_unaligned_write2(
    void *addr, bool isPtr, void *val, void *addr_src_addr,
    __sanitizer::u16 addr_src_idx, void *val_src_addr,
    __sanitizer::u16 val_src_idx);
SANITIZER_INTERFACE_ATTRIBUTE void __trec_unaligned_write4(
    void *addr, bool isPtr, void *val, void *addr_src_addr,
    __sanitizer::u16 addr_src_idx, void *val_src_addr,
    __sanitizer::u16 val_src_idx);
SANITIZER_INTERFACE_ATTRIBUTE void __trec_unaligned_write8(
    void *addr, bool isPtr, void *val, void *addr_src_addr,
    __sanitizer::u16 addr_src_idx, void *val_src_addr,
    __sanitizer::u16 val_src_idx);
#if TREC_HAS_128_BIT
SANITIZER_INTERFACE_ATTRIBUTE void __trec_unaligned_write16(
    void *addr, bool isPtr, __uint128_t val, void *addr_src_addr,
    __sanitizer::u16 addr_src_idx, void *val_src_addr,
    __sanitizer::u16 val_src_idx);
#endif
SANITIZER_INTERFACE_ATTRIBUTE void __trec_func_entry(void *);
SANITIZER_INTERFACE_ATTRIBUTE void __trec_func_exit();

#endif  // SANITIZER_GO

#ifdef __cplusplus
}  // extern "C"
#endif

namespace __trec {

// These should match declarations from public trec_interface_atomic.h
// header.
typedef unsigned char a8;
typedef unsigned short a16;
typedef unsigned int a32;
typedef unsigned long long a64;
#if !SANITIZER_GO &&                                      \
    (defined(__SIZEOF_INT128__) ||                        \
     (__clang_major__ * 100 + __clang_minor__ >= 302)) && \
    !defined(__mips64)
__extension__ typedef __int128 a128;
#define __TREC_HAS_INT128 1
#else
#define __TREC_HAS_INT128 0
#endif

// Part of ABI, do not change.
// https://github.com/llvm/llvm-project/blob/master/libcxx/include/atomic
typedef enum {
  mo_relaxed,
  mo_consume,
  mo_acquire,
  mo_release,
  mo_acq_rel,
  mo_seq_cst
} morder;

struct ThreadState;

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_load(const volatile a8 *a, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_load(const volatile a16 *a, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_load(const volatile a32 *a, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_load(const volatile a64 *a, morder mo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_load(const volatile a128 *a, morder mo, bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic8_store(volatile a8 *a, a8 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic16_store(volatile a16 *a, a16 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic32_store(volatile a32 *a, a32 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic64_store(volatile a64 *a, a64 v, morder mo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic128_store(volatile a128 *a, a128 v, morder mo, bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_exchange(volatile a8 *a, a8 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_exchange(volatile a16 *a, a16 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_exchange(volatile a32 *a, a32 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_exchange(volatile a64 *a, a64 v, morder mo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_exchange(volatile a128 *a, a128 v, morder mo, bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_add(volatile a8 *a, a8 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_add(volatile a16 *a, a16 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_add(volatile a32 *a, a32 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_add(volatile a64 *a, a64 v, morder mo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_add(volatile a128 *a, a128 v, morder mo,
                                bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_sub(volatile a8 *a, a8 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_sub(volatile a16 *a, a16 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_sub(volatile a32 *a, a32 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_sub(volatile a64 *a, a64 v, morder mo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_sub(volatile a128 *a, a128 v, morder mo,
                                bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_and(volatile a8 *a, a8 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_and(volatile a16 *a, a16 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_and(volatile a32 *a, a32 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_and(volatile a64 *a, a64 v, morder mo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_and(volatile a128 *a, a128 v, morder mo,
                                bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_or(volatile a8 *a, a8 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_or(volatile a16 *a, a16 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_or(volatile a32 *a, a32 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_or(volatile a64 *a, a64 v, morder mo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_or(volatile a128 *a, a128 v, morder mo, bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_xor(volatile a8 *a, a8 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_xor(volatile a16 *a, a16 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_xor(volatile a32 *a, a32 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_xor(volatile a64 *a, a64 v, morder mo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_xor(volatile a128 *a, a128 v, morder mo,
                                bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_nand(volatile a8 *a, a8 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_nand(volatile a16 *a, a16 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_nand(volatile a32 *a, a32 v, morder mo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_nand(volatile a64 *a, a64 v, morder mo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_nand(volatile a128 *a, a128 v, morder mo,
                                 bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic8_compare_exchange_strong(volatile a8 *a, a8 *c, a8 v,
                                           morder mo, morder fmo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic16_compare_exchange_strong(volatile a16 *a, a16 *c, a16 v,
                                            morder mo, morder fmo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic32_compare_exchange_strong(volatile a32 *a, a32 *c, a32 v,
                                            morder mo, morder fmo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic64_compare_exchange_strong(volatile a64 *a, a64 *c, a64 v,
                                            morder mo, morder fmo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic128_compare_exchange_strong(volatile a128 *a, a128 *c, a128 v,
                                             morder mo, morder fmo, bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic8_compare_exchange_weak(volatile a8 *a, a8 *c, a8 v, morder mo,
                                         morder fmo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic16_compare_exchange_weak(volatile a16 *a, a16 *c, a16 v,
                                          morder mo, morder fmo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic32_compare_exchange_weak(volatile a32 *a, a32 *c, a32 v,
                                          morder mo, morder fmo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic64_compare_exchange_weak(volatile a64 *a, a64 *c, a64 v,
                                          morder mo, morder fmo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic128_compare_exchange_weak(volatile a128 *a, a128 *c, a128 v,
                                           morder mo, morder fmo, bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_compare_exchange_val(volatile a8 *a, a8 c, a8 v, morder mo,
                                       morder fmo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_compare_exchange_val(volatile a16 *a, a16 c, a16 v,
                                         morder mo, morder fmo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_compare_exchange_val(volatile a32 *a, a32 c, a32 v,
                                         morder mo, morder fmo, bool isPtr);
SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_compare_exchange_val(volatile a64 *a, a64 c, a64 v,
                                         morder mo, morder fmo, bool isPtr);
#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_compare_exchange_val(volatile a128 *a, a128 c, a128 v,
                                           morder mo, morder fmo, bool isPtr);
#endif

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic_thread_fence(morder mo);
SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic_signal_fence(morder mo);
}  // extern "C"

}  // namespace __trec

#endif  // TREC_INTERFACE_H
