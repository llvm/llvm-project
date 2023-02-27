//===-- trec_interface_atomic.cpp
//-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
//===----------------------------------------------------------------------===//

// TraceRecorder atomic operations are based on C++11/C1x standards.
// For background see C++11 standard.  A slightly older, publicly
// available draft of the standard (not entirely up-to-date, but close enough
// for casual browsing) is available here:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3242.pdf
// The following page contains more background information:
// http://www.hpl.hp.com/personal/Hans_Boehm/c++mm/

#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "trec_flags.h"
#include "trec_interface.h"
#include "trec_rtl.h"

using namespace __trec;

#if !SANITIZER_GO && __TREC_HAS_INT128
// Protects emulation of 128-bit atomic operations.
static StaticSpinMutex mutex128;
#endif

static bool IsLoadOrder(morder mo) {
  return mo == mo_relaxed || mo == mo_consume || mo == mo_acquire ||
         mo == mo_seq_cst;
}

static bool IsStoreOrder(morder mo) {
  return mo == mo_relaxed || mo == mo_release || mo == mo_seq_cst;
}

static bool IsReleaseOrder(morder mo) {
  return mo == mo_release || mo == mo_acq_rel || mo == mo_seq_cst;
}

static bool IsAcquireOrder(morder mo) {
  return mo == mo_consume || mo == mo_acquire || mo == mo_acq_rel ||
         mo == mo_seq_cst;
}

template <typename T>
static int SizeLog() {
  if (sizeof(T) <= 1)
    return kSizeLog1;
  else if (sizeof(T) <= 2)
    return kSizeLog2;
  else if (sizeof(T) <= 4)
    return kSizeLog4;
  else
    return kSizeLog8;
  // For 16-byte atomics we also use 8-byte memory access,
  // this leads to false negatives only in very obscure cases.
}

template <typename T>
T func_xchg(volatile T *v, T op, uptr pc) {
  T res = __sync_lock_test_and_set(v, op);

  MemoryReadAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false, (uptr)res,
                   {0x8000, 1});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), true, true, false, res,
               {0, 0}, {0, (uptr)v});
  MemoryWriteAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false, (uptr)op,
                    {0x8000, 1}, {0x8000, 2});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), false, true, false, res,
               {0, 0});
  FuncExitParam(cur_thread(), (-1), 0, res);

  // __sync_lock_test_and_set does not contain full barrier.
  __sync_synchronize();
  return res;
}

template <typename T>
T func_add(volatile T *v, T op, uptr pc) {
  T res = __sync_fetch_and_add(v, op);

  MemoryReadAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false, (uptr)res,
                   {0x8000, 1});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), true, true, false, res,
               {0, 0}, {0, (uptr)v});
  MemoryWriteAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false,
                    (uptr)(res + op), {0x8000, 1}, {(u16)(op & 0x7fff), (uptr)v});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), false, true, false, res,
               {0, 0});
  FuncExitParam(cur_thread(), (-1), 0, res);

  return res;
}

template <typename T>
T func_sub(volatile T *v, T op, uptr pc) {
  T res = __sync_fetch_and_sub(v, op);

  MemoryReadAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false, (uptr)res,
                   {0x8000, 1});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), true, true, false, res,
               {0, 0}, {0, (uptr)v});
  MemoryWriteAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false,
                    (uptr)(res - op), {0x8000, 1},
                    {(u16)((-op) & 0x7fff), (uptr)v});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), false, true, false, res,
               {0, 0});
  FuncExitParam(cur_thread(), (-1), 0, res);

  return res;
}

template <typename T>
T func_and(volatile T *v, T op, uptr pc) {
  T res = __sync_fetch_and_and(v, op);

  MemoryReadAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false, (uptr)res,
                   {0x8000, 1});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), true, true, false, res,
               {0, 0}, {0, (uptr)v});
  MemoryWriteAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false,
                    (uptr)(res & op), {0x8000, 1}, {0, (uptr)v});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), false, true, false, res,
               {0, 0});
  FuncExitParam(cur_thread(), (-1), 0, res);

  return res;
}

template <typename T>
T func_or(volatile T *v, T op, uptr pc) {
  T res = __sync_fetch_and_or(v, op);

  MemoryReadAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false, (uptr)res,
                   {0x8000, 1});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), true, true, false, res,
               {0, 0}, {0, (uptr)v});
  MemoryWriteAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false,
                    (uptr)(res | op), {0x8000, 1}, {0, (uptr)v});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), false, true, false, res,
               {0, 0});
  FuncExitParam(cur_thread(), (-1), 0, res);

  return res;
}

template <typename T>
T func_xor(volatile T *v, T op, uptr pc) {
  T res = __sync_fetch_and_xor(v, op);

  MemoryReadAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false, (uptr)res,
                   {0x8000, 1});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), true, true, false, res,
               {0, 0}, {0, (uptr)v});
  MemoryWriteAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false,
                    (uptr)(res ^ op), {0x8000, 1}, {0, (uptr)v});
  MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), false, true, false, res,
               {0, 0});
  FuncExitParam(cur_thread(), (-1), 0, res);

  return res;
}

template <typename T>
T func_nand(volatile T *v, T op, uptr pc) {
  // clang does not support __sync_fetch_and_nand.
  T cmp = *v;
  for (;;) {
    T newv = ~(cmp & op);
    T cur = __sync_val_compare_and_swap(v, cmp, newv);
    if (cmp == cur) {
      MemoryReadAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false,
                       (uptr)cur, {0x8000, 1});
      MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), true, true, false, cur,
                   {0, 0}, {0, (uptr)v});
      MemoryWriteAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false,
                        (uptr)(newv), {0x8000, 1}, {0, (uptr)v});
      MemoryAccess(cur_thread(), pc, (-1), SizeLog<T>(), false, true, false,
                   cur, {0, 0});
      FuncExitParam(cur_thread(), (-1), 0, cmp);

      return cmp;
    }
  }
}

template <typename T>
T func_cas(volatile T *v, T cmp, T xch, uptr pc) {
  T res = __sync_val_compare_and_swap(v, cmp, xch);
  MemoryReadAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false, (uptr)res,
                   {0x8000, 1});
  MemoryAccess(cur_thread(), pc, -1, SizeLog<T>(), true, false, false, res,
               {0, 0}, {0, (uptr)v});
  CondBranch(cur_thread(), pc, (u64)res == cmp);
  if (res == cmp) {
    MemoryWriteAtomic(cur_thread(), pc, (uptr)v, SizeLog<T>(), false,
                      (uptr)(xch), {0x8000, 1}, {0x8000, 3});
  }
  FuncExitParam(cur_thread(), -1, 0, res);

  return res;
}

// clang does not support 128-bit atomic ops.
// Atomic ops are executed under trec internal mutex,
// here we assume that the atomic variables are not accessed
// from non-instrumented code.
#if !defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16) && !SANITIZER_GO && \
    __TREC_HAS_INT128
a128 func_xchg(volatile a128 *v, a128 op, uptr pc) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = op;
  return cmp;
}

a128 func_add(volatile a128 *v, a128 op, uptr pc) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = cmp + op;
  return cmp;
}

a128 func_sub(volatile a128 *v, a128 op, uptr pc) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = cmp - op;
  return cmp;
}

a128 func_and(volatile a128 *v, a128 op, uptr pc) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = cmp & op;
  return cmp;
}

a128 func_or(volatile a128 *v, a128 op, uptr pc) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = cmp | op;
  return cmp;
}

a128 func_xor(volatile a128 *v, a128 op, uptr pc = 0) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = cmp ^ op;
  return cmp;
}

a128 func_nand(volatile a128 *v, a128 op, uptr pc = 0) {
  SpinMutexLock lock(&mutex128);
  a128 cmp = *v;
  *v = ~(cmp & op);
  return cmp;
}

a128 func_cas(volatile a128 *v, a128 cmp, a128 xch, uptr pc) {
  SpinMutexLock lock(&mutex128);
  a128 cur = *v;
  if (cur == cmp)
    *v = xch;
  return cur;
}
#endif

#if !SANITIZER_GO
static atomic_uint8_t *to_atomic(const volatile a8 *a) {
  return reinterpret_cast<atomic_uint8_t *>(const_cast<a8 *>(a));
}

static atomic_uint16_t *to_atomic(const volatile a16 *a) {
  return reinterpret_cast<atomic_uint16_t *>(const_cast<a16 *>(a));
}
#endif

static atomic_uint32_t *to_atomic(const volatile a32 *a) {
  return reinterpret_cast<atomic_uint32_t *>(const_cast<a32 *>(a));
}

static atomic_uint64_t *to_atomic(const volatile a64 *a) {
  return reinterpret_cast<atomic_uint64_t *>(const_cast<a64 *>(a));
}

static memory_order to_mo(morder mo) {
  switch (mo) {
    case mo_relaxed:
      return memory_order_relaxed;
    case mo_consume:
      return memory_order_consume;
    case mo_acquire:
      return memory_order_acquire;
    case mo_release:
      return memory_order_release;
    case mo_acq_rel:
      return memory_order_acq_rel;
    case mo_seq_cst:
      return memory_order_seq_cst;
  }
  CHECK(0);
  return memory_order_seq_cst;
}

template <typename T>
static T NoTrecAtomicLoad(const volatile T *a, morder mo) {
  return atomic_load(to_atomic(a), to_mo(mo));
}

#if __TREC_HAS_INT128 && !SANITIZER_GO
static a128 NoTrecAtomicLoad(const volatile a128 *a, morder mo) {
  SpinMutexLock lock(&mutex128);
  return *a;
}
#endif

template <typename T>
static T AtomicLoad(ThreadState *thr, uptr pc, const volatile T *a, morder mo,
                    bool isPtr) {
  CHECK(IsLoadOrder(mo));
  // This fast-path is critical for performance.
  // Assume the access is atomic.

  if (!IsAcquireOrder(mo)) {
    T v = NoTrecAtomicLoad(a, mo);
    MemoryReadAtomic(thr, pc, (uptr)a, SizeLog<T>(), isPtr, (uptr)v,
                     {0x8000, 1});
    FuncExitParam(cur_thread(), (uptr)a, 0, v);
    return v;
  }
  // Don't create sync object if it does not exist yet. For example, an atomic
  // pointer is initialized to nullptr and then periodically acquire-loaded.
  T v = NoTrecAtomicLoad(a, mo);
  MemoryReadAtomic(thr, pc, (uptr)a, SizeLog<T>(), isPtr, (uptr)v, {0x8000, 1});
  FuncExitParam(cur_thread(), (uptr)a + 8, 0, v);
  return v;
}

template <typename T>
static void NoTrecAtomicStore(volatile T *a, T v, morder mo) {
  atomic_store(to_atomic(a), v, to_mo(mo));
}

#if __TREC_HAS_INT128 && !SANITIZER_GO
static void NoTrecAtomicStore(volatile a128 *a, a128 v, morder mo) {
  SpinMutexLock lock(&mutex128);
  *a = v;
}
#endif

template <typename T>
static void AtomicStore(ThreadState *thr, uptr pc, volatile T *a, T v,
                        morder mo, bool isPtr) {
  CHECK(IsStoreOrder(mo));
  MemoryWriteAtomic(thr, pc, (uptr)a, SizeLog<T>(), isPtr, (uptr)v, {0x8000, 1},
                    {0x8000, 2});

  // This fast-path is critical for performance.
  // Assume the access is atomic.
  // Strictly saying even relaxed store cuts off release sequence,
  // so must reset the clock.
  if (!IsReleaseOrder(mo)) {
    NoTrecAtomicStore(a, v, mo);
    return;
  }
  __sync_synchronize();
  NoTrecAtomicStore(a, v, mo);
}

template <typename T, T (*F)(volatile T *v, T op, uptr pc)>
static T AtomicRMW(ThreadState *thr, uptr pc, volatile T *a, T v, morder mo,
                   bool isPtr) {
  v = F(a, v, pc);
  return v;
}

template <typename T>
static T NoTrecAtomicExchange(volatile T *a, T v, morder mo) {
  return func_xchg(a, v, StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()));
}

template <typename T>
static T NoTrecAtomicFetchAdd(volatile T *a, T v, morder mo) {
  return func_add(a, v, StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()));
}

template <typename T>
static T NoTrecAtomicFetchSub(volatile T *a, T v, morder mo) {
  return func_sub(a, v, StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()));
}

template <typename T>
static T NoTrecAtomicFetchAnd(volatile T *a, T v, morder mo) {
  return func_and(a, v, StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()));
}

template <typename T>
static T NoTrecAtomicFetchOr(volatile T *a, T v, morder mo) {
  return func_or(a, v, StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()));
}

template <typename T>
static T NoTrecAtomicFetchXor(volatile T *a, T v, morder mo) {
  return func_xor(a, v, StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()));
}

template <typename T>
static T NoTrecAtomicFetchNand(volatile T *a, T v, morder mo) {
  return func_nand(a, v, StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()));
}

template <typename T>
static T AtomicExchange(ThreadState *thr, uptr pc, volatile T *a, T v,
                        morder mo, bool isPtr) {
  T res = AtomicRMW<T, func_xchg>(thr, pc, a, v, mo, isPtr);
  return res;
}

template <typename T>
static T AtomicFetchAdd(ThreadState *thr, uptr pc, volatile T *a, T v,
                        morder mo, bool isPtr) {
  T res = AtomicRMW<T, func_add>(thr, pc, a, v, mo, isPtr);
  return res;
}

template <typename T>
static T AtomicFetchSub(ThreadState *thr, uptr pc, volatile T *a, T v,
                        morder mo, bool isPtr) {
  T res = AtomicRMW<T, func_sub>(thr, pc, a, v, mo, isPtr);
  return res;
}

template <typename T>
static T AtomicFetchAnd(ThreadState *thr, uptr pc, volatile T *a, T v,
                        morder mo, bool isPtr) {
  T res = AtomicRMW<T, func_and>(thr, pc, a, v, mo, isPtr);
  return res;
}

template <typename T>
static T AtomicFetchOr(ThreadState *thr, uptr pc, volatile T *a, T v, morder mo,
                       bool isPtr) {
  T res = AtomicRMW<T, func_or>(thr, pc, a, v, mo, isPtr);
  return res;
}

template <typename T>
static T AtomicFetchXor(ThreadState *thr, uptr pc, volatile T *a, T v,
                        morder mo, bool isPtr) {
  T res = AtomicRMW<T, func_xor>(thr, pc, a, v, mo, isPtr);
  return res;
}

template <typename T>
static T AtomicFetchNand(ThreadState *thr, uptr pc, volatile T *a, T v,
                         morder mo, bool isPtr) {
  T res = AtomicRMW<T, func_nand>(thr, pc, a, v, mo, isPtr);
  return res;
}

template <typename T>
static bool NoTrecAtomicCAS(volatile T *a, T *c, T v, morder mo, morder fmo) {
  return atomic_compare_exchange_strong(to_atomic(a), c, v, to_mo(mo));
}

#if __TREC_HAS_INT128
static bool NoTrecAtomicCAS(volatile a128 *a, a128 *c, a128 v, morder mo,
                            morder fmo) {
  a128 old = *c;
  a128 cur = func_cas(a, old, v, 0);
  if (cur == old)
    return true;
  *c = cur;
  return false;
}
#endif

template <typename T>
static T NoTrecAtomicCAS(volatile T *a, T c, T v, morder mo, morder fmo) {
  NoTrecAtomicCAS(a, &c, v, mo, fmo);
  return c;
}

template <typename T>
static bool AtomicCAS(ThreadState *thr, uptr pc, volatile T *a, T *c, T v,
                      morder mo, morder fmo, bool isPtr) {
  (void)fmo;  // Unused because llvm does not pass it yet.
  T cc = *c;
  T pr = func_cas(a, cc, v, pc);
  if (pr == cc) {
    return true;
  }
  *c = pr;
  return false;
}

template <typename T>
static T AtomicCAS(ThreadState *thr, uptr pc, volatile T *a, T c, T v,
                   morder mo, morder fmo, bool isPtr) {
  AtomicCAS(thr, pc, a, &c, v, mo, fmo, isPtr);
  return c;
}

#if !SANITIZER_GO
static void NoTrecAtomicFence(morder mo) { __sync_synchronize(); }

static void AtomicFence(ThreadState *thr, uptr pc, morder mo) {
  // FIXME(dvyukov): not implemented.
  __sync_synchronize();
}
#endif

// Interface functions follow.
#if !SANITIZER_GO

// C/C++

static morder convert_morder(morder mo) {
  if (flags()->force_seq_cst_atomics)
    return (morder)mo_seq_cst;

  // Filter out additional memory order flags:
  // MEMMODEL_SYNC        = 1 << 15
  // __ATOMIC_HLE_ACQUIRE = 1 << 16
  // __ATOMIC_HLE_RELEASE = 1 << 17
  //
  // HLE is an optimization, and we pretend that elision always fails.
  // MEMMODEL_SYNC is used when lowering __sync_ atomics,
  // since we use __sync_ atomics for actual atomic operations,
  // we can safely ignore it as well. It also subtly affects semantics,
  // but we don't model the difference.
  return (morder)(mo & 0x7fff);
}

#define SCOPED_ATOMIC(func, ...)                                \
  ThreadState *const thr = cur_thread();                        \
  if (UNLIKELY(thr->ignore_sync || thr->ignore_interceptors)) { \
    ProcessPendingSignals(thr);                                 \
    return NoTrecAtomic##func(__VA_ARGS__);                     \
  }                                                             \
  const uptr callpc = (uptr)__builtin_return_address(0);        \
  mo = convert_morder(mo);                                      \
  ScopedAtomic sa(thr, callpc, a, mo, __func__);                \
  return Atomic##func(thr, callpc, __VA_ARGS__);                \
  /**/

#define SCOPED_ATOMIC_TREC(func, isPtr, ...)                    \
  ThreadState *const thr = cur_thread();                        \
  if (UNLIKELY(thr->ignore_sync || thr->ignore_interceptors)) { \
    ProcessPendingSignals(thr);                                 \
    return NoTrecAtomic##func(__VA_ARGS__);                     \
  }                                                             \
  const uptr callpc = (uptr)__builtin_return_address(0);        \
  mo = convert_morder(mo);                                      \
  ScopedAtomic sa(thr, callpc, a, mo, __func__);                \
  return Atomic##func(thr, callpc, __VA_ARGS__, isPtr);         \
  /**/

class ScopedAtomic {
 public:
  ScopedAtomic(ThreadState *thr, uptr pc, const volatile void *a, morder mo,
               const char *func)
      : thr_(thr) {
    DPrintf("#%d: %s(%p, %d)\n", thr_->tid, func, a, mo);
  }
  ~ScopedAtomic() { ProcessPendingSignals(thr_); }

 private:
  ThreadState *thr_;
};

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_load(const volatile a8 *a, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Load, isPtr, a, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_load(const volatile a16 *a, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Load, isPtr, a, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_load(const volatile a32 *a, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Load, isPtr, a, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_load(const volatile a64 *a, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Load, isPtr, a, mo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_load(const volatile a128 *a, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Load, isPtr, a, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic8_store(volatile a8 *a, a8 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Store, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic16_store(volatile a16 *a, a16 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Store, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic32_store(volatile a32 *a, a32 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Store, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic64_store(volatile a64 *a, a64 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Store, isPtr, a, v, mo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic128_store(volatile a128 *a, a128 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Store, isPtr, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_exchange(volatile a8 *a, a8 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Exchange, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_exchange(volatile a16 *a, a16 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Exchange, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_exchange(volatile a32 *a, a32 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Exchange, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_exchange(volatile a64 *a, a64 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(Exchange, isPtr, a, v, mo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_exchange(volatile a128 *a, a128 v, morder mo,
                               bool isPtr) {
  SCOPED_ATOMIC_TREC(Exchange, isPtr, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_add(volatile a8 *a, a8 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchAdd, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_add(volatile a16 *a, a16 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchAdd, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_add(volatile a32 *a, a32 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchAdd, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_add(volatile a64 *a, a64 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchAdd, isPtr, a, v, mo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_add(volatile a128 *a, a128 v, morder mo,
                                bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchAdd, isPtr, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_sub(volatile a8 *a, a8 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchSub, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_sub(volatile a16 *a, a16 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchSub, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_sub(volatile a32 *a, a32 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchSub, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_sub(volatile a64 *a, a64 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchSub, isPtr, a, v, mo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_sub(volatile a128 *a, a128 v, morder mo,
                                bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchSub, isPtr, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_and(volatile a8 *a, a8 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchAnd, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_and(volatile a16 *a, a16 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchAnd, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_and(volatile a32 *a, a32 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchAnd, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_and(volatile a64 *a, a64 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchAnd, isPtr, a, v, mo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_and(volatile a128 *a, a128 v, morder mo,
                                bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchAnd, isPtr, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_or(volatile a8 *a, a8 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchOr, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_or(volatile a16 *a, a16 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchOr, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_or(volatile a32 *a, a32 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchOr, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_or(volatile a64 *a, a64 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchOr, isPtr, a, v, mo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_or(volatile a128 *a, a128 v, morder mo,
                               bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchOr, isPtr, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_xor(volatile a8 *a, a8 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchXor, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_xor(volatile a16 *a, a16 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchXor, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_xor(volatile a32 *a, a32 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchXor, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_xor(volatile a64 *a, a64 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchXor, isPtr, a, v, mo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_xor(volatile a128 *a, a128 v, morder mo,
                                bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchXor, isPtr, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_fetch_nand(volatile a8 *a, a8 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchNand, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_fetch_nand(volatile a16 *a, a16 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchNand, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_fetch_nand(volatile a32 *a, a32 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchNand, isPtr, a, v, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_fetch_nand(volatile a64 *a, a64 v, morder mo, bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchNand, isPtr, a, v, mo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_fetch_nand(volatile a128 *a, a128 v, morder mo,
                                 bool isPtr) {
  SCOPED_ATOMIC_TREC(FetchNand, isPtr, a, v, mo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic8_compare_exchange_strong(volatile a8 *a, a8 *c, a8 v,
                                           morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic16_compare_exchange_strong(volatile a16 *a, a16 *c, a16 v,
                                            morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic32_compare_exchange_strong(volatile a32 *a, a32 *c, a32 v,
                                            morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic64_compare_exchange_strong(volatile a64 *a, a64 *c, a64 v,
                                            morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic128_compare_exchange_strong(volatile a128 *a, a128 *c, a128 v,
                                             morder mo, morder fmo,
                                             bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic8_compare_exchange_weak(volatile a8 *a, a8 *c, a8 v, morder mo,
                                         morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic16_compare_exchange_weak(volatile a16 *a, a16 *c, a16 v,
                                          morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic32_compare_exchange_weak(volatile a32 *a, a32 *c, a32 v,
                                          morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic64_compare_exchange_weak(volatile a64 *a, a64 *c, a64 v,
                                          morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
int __trec_atomic128_compare_exchange_weak(volatile a128 *a, a128 *c, a128 v,
                                           morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
a8 __trec_atomic8_compare_exchange_val(volatile a8 *a, a8 c, a8 v, morder mo,
                                       morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a16 __trec_atomic16_compare_exchange_val(volatile a16 *a, a16 c, a16 v,
                                         morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a32 __trec_atomic32_compare_exchange_val(volatile a32 *a, a32 c, a32 v,
                                         morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

SANITIZER_INTERFACE_ATTRIBUTE
a64 __trec_atomic64_compare_exchange_val(volatile a64 *a, a64 c, a64 v,
                                         morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}

#if __TREC_HAS_INT128
SANITIZER_INTERFACE_ATTRIBUTE
a128 __trec_atomic128_compare_exchange_val(volatile a128 *a, a128 c, a128 v,
                                           morder mo, morder fmo, bool isPtr) {
  SCOPED_ATOMIC_TREC(CAS, isPtr, a, c, v, mo, fmo);
}
#endif

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic_thread_fence(morder mo) {
  char *a = 0;
  SCOPED_ATOMIC(Fence, mo);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_atomic_signal_fence(morder mo) {}
}  // extern "C"

#else  // #if !SANITIZER_GO

// Go

#define ATOMIC(func, ...)               \
  if (thr->ignore_sync) {               \
    NoTrecAtomic##func(__VA_ARGS__);    \
  } else {                              \
    Atomic##func(thr, pc, __VA_ARGS__); \
  }                                     \
  /**/

#define ATOMIC_RET(func, ret, ...)              \
  if (thr->ignore_sync) {                       \
    (ret) = NoTrecAtomic##func(__VA_ARGS__);    \
  } else {                                      \
    (ret) = Atomic##func(thr, pc, __VA_ARGS__); \
  }                                             \
/**/

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __trec_go_atomic32_load(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(Load, *(a32 *)(a + 8), *(a32 **)a, mo_acquire);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_go_atomic64_load(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(Load, *(a64 *)(a + 8), *(a64 **)a, mo_acquire);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_go_atomic32_store(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC(Store, *(a32 **)a, *(a32 *)(a + 8), mo_release);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_go_atomic64_store(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC(Store, *(a64 **)a, *(a64 *)(a + 8), mo_release);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_go_atomic32_fetch_add(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(FetchAdd, *(a32 *)(a + 16), *(a32 **)a, *(a32 *)(a + 8),
             mo_acq_rel);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_go_atomic64_fetch_add(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(FetchAdd, *(a64 *)(a + 16), *(a64 **)a, *(a64 *)(a + 8),
             mo_acq_rel);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_go_atomic32_exchange(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(Exchange, *(a32 *)(a + 16), *(a32 **)a, *(a32 *)(a + 8),
             mo_acq_rel);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_go_atomic64_exchange(ThreadState *thr, uptr cpc, uptr pc, u8 *a) {
  ATOMIC_RET(Exchange, *(a64 *)(a + 16), *(a64 **)a, *(a64 *)(a + 8),
             mo_acq_rel);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_go_atomic32_compare_exchange(ThreadState *thr, uptr cpc, uptr pc,
                                         u8 *a) {
  a32 cur = 0;
  a32 cmp = *(a32 *)(a + 8);
  ATOMIC_RET(CAS, cur, *(a32 **)a, cmp, *(a32 *)(a + 12), mo_acq_rel,
             mo_acquire);
  *(bool *)(a + 16) = (cur == cmp);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __trec_go_atomic64_compare_exchange(ThreadState *thr, uptr cpc, uptr pc,
                                         u8 *a) {
  a64 cur = 0;
  a64 cmp = *(a64 *)(a + 8);
  ATOMIC_RET(CAS, cur, *(a64 **)a, cmp, *(a64 *)(a + 16), mo_acq_rel,
             mo_acquire);
  *(bool *)(a + 24) = (cur == cmp);
}
}  // extern "C"
#endif  // #if !SANITIZER_GO
