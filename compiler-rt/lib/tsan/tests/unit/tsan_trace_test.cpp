//===-- tsan_trace_test.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#include "tsan_trace.h"

#include <pthread.h>

#include "gtest/gtest.h"
#include "tsan_rtl.h"

#if SANITIZER_MAC || !defined(__x86_64__)
// These tests are currently crashing on Mac:
// https://reviews.llvm.org/D107911
// and on ppc64: https://reviews.llvm.org/D110546#3025422
// due to the way we create thread contexts
// (but they crashed on Mac with normal pthread_create as well).
// There must be some difference in thread initialization
// between normal execution and unit tests.
#  define TRACE_TEST(SUITE, NAME) TEST(SUITE, DISABLED_##NAME)
#else
#  define TRACE_TEST(SUITE, NAME) TEST(SUITE, NAME)
#endif

namespace __tsan {

using namespace v3;

// We need to run all trace tests in a new thread,
// so that the thread trace is empty initially.
template <uptr N>
struct ThreadArray {
  ThreadArray() {
    for (auto *&thr : threads) {
      thr = static_cast<ThreadState *>(
          MmapOrDie(sizeof(ThreadState), "ThreadState"));
      Tid tid = ThreadCreate(cur_thread(), 0, 0, true);
      Processor *proc = ProcCreate();
      ProcWire(proc, thr);
      ThreadStart(thr, tid, 0, ThreadType::Fiber);
    }
  }

  ~ThreadArray() {
    for (uptr i = 0; i < N; i++) {
      if (threads[i])
        Finish(i);
    }
  }

  void Finish(uptr i) {
    auto *thr = threads[i];
    threads[i] = nullptr;
    Processor *proc = thr->proc();
    ThreadFinish(thr);
    ProcUnwire(proc, thr);
    ProcDestroy(proc);
    UnmapOrDie(thr, sizeof(ThreadState));
  }

  ThreadState *threads[N];
  ThreadState *operator[](uptr i) { return threads[i]; }
  ThreadState *operator->() { return threads[0]; }
  operator ThreadState *() { return threads[0]; }
};

TRACE_TEST(Trace, RestoreAccess) {
  // A basic test with some function entry/exit events,
  // some mutex lock/unlock events and some other distracting
  // memory events.
  ThreadArray<1> thr;
  TraceFunc(thr, 0x1000);
  TraceFunc(thr, 0x1001);
  TraceMutexLock(thr, v3::EventType::kLock, 0x4000, 0x5000, 0x6000);
  TraceMutexLock(thr, v3::EventType::kLock, 0x4001, 0x5001, 0x6001);
  TraceMutexUnlock(thr, 0x5000);
  TraceFunc(thr);
  CHECK(TryTraceMemoryAccess(thr, 0x2001, 0x3001, 8, kAccessRead));
  TraceMutexLock(thr, v3::EventType::kRLock, 0x4002, 0x5002, 0x6002);
  TraceFunc(thr, 0x1002);
  CHECK(TryTraceMemoryAccess(thr, 0x2000, 0x3000, 8, kAccessRead));
  // This is the access we want to find.
  // The previous one is equivalent, but RestoreStack must prefer
  // the last of the matchig accesses.
  CHECK(TryTraceMemoryAccess(thr, 0x2002, 0x3000, 8, kAccessRead));
  Lock lock1(&ctx->slot_mtx);
  ThreadRegistryLock lock2(&ctx->thread_registry);
  VarSizeStackTrace stk;
  MutexSet mset;
  uptr tag = kExternalTagNone;
  bool res =
      RestoreStack(thr->tid, v3::EventType::kAccessExt, thr->sid, thr->epoch,
                   0x3000, 8, kAccessRead, &stk, &mset, &tag);
  CHECK(res);
  CHECK_EQ(stk.size, 3);
  CHECK_EQ(stk.trace[0], 0x1000);
  CHECK_EQ(stk.trace[1], 0x1002);
  CHECK_EQ(stk.trace[2], 0x2002);
  CHECK_EQ(mset.Size(), 2);
  CHECK_EQ(mset.Get(0).addr, 0x5001);
  CHECK_EQ(mset.Get(0).stack_id, 0x6001);
  CHECK_EQ(mset.Get(0).write, true);
  CHECK_EQ(mset.Get(1).addr, 0x5002);
  CHECK_EQ(mset.Get(1).stack_id, 0x6002);
  CHECK_EQ(mset.Get(1).write, false);
  CHECK_EQ(tag, kExternalTagNone);
}

TRACE_TEST(Trace, MemoryAccessSize) {
  // Test tracing and matching of accesses of different sizes.
  struct Params {
    uptr access_size, offset, size;
    bool res;
  };
  Params tests[] = {
      {1, 0, 1, true},  {4, 0, 2, true},
      {4, 2, 2, true},  {8, 3, 1, true},
      {2, 1, 1, true},  {1, 1, 1, false},
      {8, 5, 4, false}, {4, static_cast<uptr>(-1l), 4, false},
  };
  for (auto params : tests) {
    for (int type = 0; type < 3; type++) {
      ThreadArray<1> thr;
      Printf("access_size=%zu, offset=%zu, size=%zu, res=%d, type=%d\n",
             params.access_size, params.offset, params.size, params.res, type);
      TraceFunc(thr, 0x1000);
      switch (type) {
        case 0:
          // This should emit compressed event.
          CHECK(TryTraceMemoryAccess(thr, 0x2000, 0x3000, params.access_size,
                                     kAccessRead));
          break;
        case 1:
          // This should emit full event.
          CHECK(TryTraceMemoryAccess(thr, 0x2000000, 0x3000, params.access_size,
                                     kAccessRead));
          break;
        case 2:
          TraceMemoryAccessRange(thr, 0x2000000, 0x3000, params.access_size,
                                 kAccessRead);
          break;
      }
      Lock lock1(&ctx->slot_mtx);
      ThreadRegistryLock lock2(&ctx->thread_registry);
      VarSizeStackTrace stk;
      MutexSet mset;
      uptr tag = kExternalTagNone;
      bool res = RestoreStack(thr->tid, v3::EventType::kAccessExt, thr->sid,
                              thr->epoch, 0x3000 + params.offset, params.size,
                              kAccessRead, &stk, &mset, &tag);
      CHECK_EQ(res, params.res);
      if (params.res) {
        CHECK_EQ(stk.size, 2);
        CHECK_EQ(stk.trace[0], 0x1000);
        CHECK_EQ(stk.trace[1], type ? 0x2000000 : 0x2000);
      }
    }
  }
}

TRACE_TEST(Trace, RestoreMutexLock) {
  // Check of restoration of a mutex lock event.
  ThreadArray<1> thr;
  TraceFunc(thr, 0x1000);
  TraceMutexLock(thr, v3::EventType::kLock, 0x4000, 0x5000, 0x6000);
  TraceMutexLock(thr, v3::EventType::kRLock, 0x4001, 0x5001, 0x6001);
  TraceMutexLock(thr, v3::EventType::kRLock, 0x4002, 0x5001, 0x6002);
  Lock lock1(&ctx->slot_mtx);
  ThreadRegistryLock lock2(&ctx->thread_registry);
  VarSizeStackTrace stk;
  MutexSet mset;
  uptr tag = kExternalTagNone;
  bool res = RestoreStack(thr->tid, v3::EventType::kLock, thr->sid, thr->epoch,
                          0x5001, 0, 0, &stk, &mset, &tag);
  CHECK(res);
  CHECK_EQ(stk.size, 2);
  CHECK_EQ(stk.trace[0], 0x1000);
  CHECK_EQ(stk.trace[1], 0x4002);
  CHECK_EQ(mset.Size(), 2);
  CHECK_EQ(mset.Get(0).addr, 0x5000);
  CHECK_EQ(mset.Get(0).stack_id, 0x6000);
  CHECK_EQ(mset.Get(0).write, true);
  CHECK_EQ(mset.Get(1).addr, 0x5001);
  CHECK_EQ(mset.Get(1).stack_id, 0x6001);
  CHECK_EQ(mset.Get(1).write, false);
}

TRACE_TEST(Trace, MultiPart) {
  // Check replay of a trace with multiple parts.
  ThreadArray<1> thr;
  TraceFunc(thr, 0x1000);
  TraceFunc(thr, 0x2000);
  TraceMutexLock(thr, v3::EventType::kLock, 0x4000, 0x5000, 0x6000);
  const uptr kEvents = 3 * sizeof(TracePart) / sizeof(v3::Event);
  for (uptr i = 0; i < kEvents; i++) {
    TraceFunc(thr, 0x3000);
    TraceMutexLock(thr, v3::EventType::kLock, 0x4002, 0x5002, 0x6002);
    TraceMutexUnlock(thr, 0x5002);
    TraceFunc(thr);
  }
  TraceFunc(thr, 0x4000);
  TraceMutexLock(thr, v3::EventType::kRLock, 0x4001, 0x5001, 0x6001);
  CHECK(TryTraceMemoryAccess(thr, 0x2002, 0x3000, 8, kAccessRead));
  Lock lock1(&ctx->slot_mtx);
  ThreadRegistryLock lock2(&ctx->thread_registry);
  VarSizeStackTrace stk;
  MutexSet mset;
  uptr tag = kExternalTagNone;
  bool res =
      RestoreStack(thr->tid, v3::EventType::kAccessExt, thr->sid, thr->epoch,
                   0x3000, 8, kAccessRead, &stk, &mset, &tag);
  CHECK(res);
  CHECK_EQ(stk.size, 4);
  CHECK_EQ(stk.trace[0], 0x1000);
  CHECK_EQ(stk.trace[1], 0x2000);
  CHECK_EQ(stk.trace[2], 0x4000);
  CHECK_EQ(stk.trace[3], 0x2002);
  CHECK_EQ(mset.Size(), 2);
  CHECK_EQ(mset.Get(0).addr, 0x5000);
  CHECK_EQ(mset.Get(0).stack_id, 0x6000);
  CHECK_EQ(mset.Get(0).write, true);
  CHECK_EQ(mset.Get(1).addr, 0x5001);
  CHECK_EQ(mset.Get(1).stack_id, 0x6001);
  CHECK_EQ(mset.Get(1).write, false);
}

}  // namespace __tsan
