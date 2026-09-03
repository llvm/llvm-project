//===-- dsan_mac.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer.
//
// Mac-specific details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_APPLE

#  include <pthread.h>

#  include "dsan.h"
#  include "dsan_allocator.h"
#  include "dsan_thread.h"
#  include "interception/interception.h"
#  include "sanitizer_common/sanitizer_allocator_internal.h"

namespace __dsan {
struct ThreadLocalData {
  ThreadContextDsanBase* current_thread;
  AllocatorCache cache;
};

static pthread_key_t thread_local_key;
static pthread_once_t thread_local_key_once = PTHREAD_ONCE_INIT;

static void RestoreThreadLocalData(void* ptr) {
  ThreadLocalData* data = static_cast<ThreadLocalData*>(ptr);
  if (data->current_thread)
    pthread_setspecific(thread_local_key, data);
}

static void CreateThreadLocalKey() {
  CHECK_EQ(pthread_key_create(&thread_local_key, RestoreThreadLocalData), 0);
}

static ThreadLocalData* GetThreadLocalData(bool allocate) {
  pthread_once(&thread_local_key_once, CreateThreadLocalKey);
  ThreadLocalData* data =
      static_cast<ThreadLocalData*>(pthread_getspecific(thread_local_key));
  if (!data && allocate) {
    data = static_cast<ThreadLocalData*>(InternalAlloc(sizeof(*data)));
    data->current_thread = nullptr;
    data->cache = AllocatorCache();
    pthread_setspecific(thread_local_key, data);
  }
  return data;
}

ThreadContextDsanBase* GetCurrentThread() {
  ThreadLocalData* data = GetThreadLocalData(false);
  return data ? data->current_thread : nullptr;
}

void SetCurrentThread(ThreadContextDsanBase* tctx) {
  GetThreadLocalData(true)->current_thread = tctx;
}

AllocatorCache* GetAllocatorCache() { return &GetThreadLocalData(true)->cache; }

// Support for the following functions from libdispatch on Mac OS:
//   dispatch_async_f()
//   dispatch_async()
//   dispatch_sync_f()
//   dispatch_sync()
//   dispatch_after_f()
//   dispatch_after()
//   dispatch_group_async_f()
//   dispatch_group_async()
// TODO(glider): libdispatch API contains other functions that we don't support
// yet.
//
// dispatch_sync() and dispatch_sync_f() are synchronous, although chances are
// they can cause jobs to run on a thread different from the current one.
// TODO(glider): if so, we need a test for this (otherwise we should remove
// them).
//
// The following functions use dispatch_barrier_async_f() (which isn't a library
// function but is exported) and are thus supported:
//   dispatch_source_set_cancel_handler_f()
//   dispatch_source_set_cancel_handler()
//   dispatch_source_set_event_handler_f()
//   dispatch_source_set_event_handler()
//
// The reference manual for Grand Central Dispatch is available at
//   http://developer.apple.com/library/mac/#documentation/Performance/Reference/GCD_libdispatch_Ref/Reference/reference.html
// The implementation details are at
//   http://libdispatch.macosforge.org/trac/browser/trunk/src/queue.c

typedef void* dispatch_group_t;
typedef void* dispatch_queue_t;
typedef void* dispatch_source_t;
typedef u64 dispatch_time_t;
typedef void (*dispatch_function_t)(void* block);
typedef void* (*worker_t)(void* block);

// A wrapper for the ObjC blocks used to support libdispatch.
typedef struct {
  void* block;
  dispatch_function_t func;
  u32 parent_tid;
} dsan_block_context_t;

ALWAYS_INLINE
void dsan_register_worker_thread(int parent_tid) {
  if (GetCurrentThreadId() == kInvalidTid) {
    u32 tid = ThreadCreate(parent_tid, true);
    ThreadStart(tid, GetTid());
  }
}

// For use by only those functions that allocated the context via
// alloc_dsan_context().
extern "C" void dsan_dispatch_call_block_and_release(void* block) {
  dsan_block_context_t* context = (dsan_block_context_t*)block;
  VReport(2,
          "dsan_dispatch_call_block_and_release(): "
          "context: %p, pthread_self: %p\n",
          block, (void*)pthread_self());
  dsan_register_worker_thread(context->parent_tid);
  // Call the original dispatcher for the block.
  context->func(context->block);
  GET_STACK_TRACE_MALLOC;
  dsan_free(context, stack);
}

}  // namespace __dsan

using namespace __dsan;

// Wrap |ctxt| and |func| into an dsan_block_context_t.
// The caller retains control of the allocated context.
extern "C" dsan_block_context_t* alloc_dsan_context(void* ctxt,
                                                    dispatch_function_t func) {
  GET_STACK_TRACE_THREAD;
  dsan_block_context_t* dsan_ctxt =
      (dsan_block_context_t*)dsan_malloc(sizeof(dsan_block_context_t), stack);
  dsan_ctxt->block = ctxt;
  dsan_ctxt->func = func;
  dsan_ctxt->parent_tid = GetCurrentThreadId();
  return dsan_ctxt;
}

// Define interceptor for dispatch_*_f function with the three most common
// parameters: dispatch_queue_t, context, dispatch_function_t.
#  define INTERCEPT_DISPATCH_X_F_3(dispatch_x_f)                        \
    INTERCEPTOR(void, dispatch_x_f, dispatch_queue_t dq, void* ctxt,    \
                dispatch_function_t func) {                             \
      dsan_block_context_t* dsan_ctxt = alloc_dsan_context(ctxt, func); \
      return REAL(dispatch_x_f)(dq, (void*)dsan_ctxt,                   \
                                dsan_dispatch_call_block_and_release);  \
    }

INTERCEPT_DISPATCH_X_F_3(dispatch_async_f)
INTERCEPT_DISPATCH_X_F_3(dispatch_sync_f)
INTERCEPT_DISPATCH_X_F_3(dispatch_barrier_async_f)

INTERCEPTOR(void, dispatch_after_f, dispatch_time_t when, dispatch_queue_t dq,
            void* ctxt, dispatch_function_t func) {
  dsan_block_context_t* dsan_ctxt = alloc_dsan_context(ctxt, func);
  return REAL(dispatch_after_f)(when, dq, (void*)dsan_ctxt,
                                dsan_dispatch_call_block_and_release);
}

INTERCEPTOR(void, dispatch_group_async_f, dispatch_group_t group,
            dispatch_queue_t dq, void* ctxt, dispatch_function_t func) {
  dsan_block_context_t* dsan_ctxt = alloc_dsan_context(ctxt, func);
  REAL(dispatch_group_async_f)(group, dq, (void*)dsan_ctxt,
                               dsan_dispatch_call_block_and_release);
}

#  if !defined(MISSING_BLOCKS_SUPPORT)
extern "C" {
void dispatch_async(dispatch_queue_t dq, void (^work)(void));
void dispatch_group_async(dispatch_group_t dg, dispatch_queue_t dq,
                          void (^work)(void));
void dispatch_after(dispatch_time_t when, dispatch_queue_t queue,
                    void (^work)(void));
void dispatch_source_set_cancel_handler(dispatch_source_t ds,
                                        void (^work)(void));
void dispatch_source_set_event_handler(dispatch_source_t ds,
                                       void (^work)(void));
}

#    define GET_DSAN_BLOCK(work)                 \
      void (^dsan_block)(void);                  \
      int parent_tid = GetCurrentThreadId();     \
      dsan_block = ^(void) {                     \
        dsan_register_worker_thread(parent_tid); \
        work();                                  \
      }

INTERCEPTOR(void, dispatch_async, dispatch_queue_t dq, void (^work)(void)) {
  GET_DSAN_BLOCK(work);
  REAL(dispatch_async)(dq, dsan_block);
}

INTERCEPTOR(void, dispatch_group_async, dispatch_group_t dg,
            dispatch_queue_t dq, void (^work)(void)) {
  GET_DSAN_BLOCK(work);
  REAL(dispatch_group_async)(dg, dq, dsan_block);
}

INTERCEPTOR(void, dispatch_after, dispatch_time_t when, dispatch_queue_t queue,
            void (^work)(void)) {
  GET_DSAN_BLOCK(work);
  REAL(dispatch_after)(when, queue, dsan_block);
}

INTERCEPTOR(void, dispatch_source_set_cancel_handler, dispatch_source_t ds,
            void (^work)(void)) {
  if (!work) {
    REAL(dispatch_source_set_cancel_handler)(ds, work);
    return;
  }
  GET_DSAN_BLOCK(work);
  REAL(dispatch_source_set_cancel_handler)(ds, dsan_block);
}

INTERCEPTOR(void, dispatch_source_set_event_handler, dispatch_source_t ds,
            void (^work)(void)) {
  GET_DSAN_BLOCK(work);
  REAL(dispatch_source_set_event_handler)(ds, dsan_block);
}
#  endif

#endif  // SANITIZER_APPLE
