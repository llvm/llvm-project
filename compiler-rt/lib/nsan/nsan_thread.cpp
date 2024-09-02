//===- nsan_threads.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Thread management.
//===----------------------------------------------------------------------===//

#include "nsan_thread.h"

#include <pthread.h>

#include "nsan.h"
#include "sanitizer_common/sanitizer_tls_get_addr.h"

using namespace __nsan;

NsanThread *NsanThread::Create(thread_callback_t start_routine, void *arg) {
  uptr PageSize = GetPageSizeCached();
  uptr size = RoundUpTo(sizeof(NsanThread), PageSize);
  NsanThread *thread = (NsanThread *)MmapOrDie(size, __func__);
  thread->start_routine_ = start_routine;
  thread->arg_ = arg;
  thread->destructor_iterations_ = GetPthreadDestructorIterations();

  return thread;
}

void NsanThread::SetThreadStackAndTls() {
  uptr tls_size = 0;
  uptr stack_size = 0;
  GetThreadStackAndTls(IsMainThread(), &stack_.bottom, &stack_size, &tls_begin_,
                       &tls_size);
  stack_.top = stack_.bottom + stack_size;
  tls_end_ = tls_begin_ + tls_size;

  int local;
  CHECK(AddrIsInStack((uptr)&local));
}

void NsanThread::ClearShadowForThreadStackAndTLS() {
  __nsan_set_value_unknown((const u8 *)stack_.bottom,
                           stack_.top - stack_.bottom);
  if (tls_begin_ != tls_end_)
    __nsan_set_value_unknown((const u8 *)tls_begin_, tls_end_ - tls_begin_);
  DTLS *dtls = DTLS_Get();
  CHECK_NE(dtls, 0);
  ForEachDVT(dtls, [](const DTLS::DTV &dtv, int id) {
    __nsan_set_value_unknown((const u8 *)dtv.beg, dtv.size);
  });
}

void NsanThread::Init() {
  SetThreadStackAndTls();
  ClearShadowForThreadStackAndTLS();
  malloc_storage().Init();
}

void NsanThread::TSDDtor(void *tsd) {
  NsanThread *t = (NsanThread *)tsd;
  t->Destroy();
}

void NsanThread::Destroy() {
  malloc_storage().CommitBack();
  // We also clear the shadow on thread destruction because
  // some code may still be executing in later TSD destructors
  // and we don't want it to have any poisoned stack.
  ClearShadowForThreadStackAndTLS();
  uptr size = RoundUpTo(sizeof(NsanThread), GetPageSizeCached());
  UnmapOrDie(this, size);
  DTLS_Destroy();
}

thread_return_t NsanThread::ThreadStart() {
  if (!start_routine_) {
    // start_routine_ == 0 if we're on the main thread or on one of the
    // OS X libdispatch worker threads. But nobody is supposed to call
    // ThreadStart() for the worker threads.
    return 0;
  }

  return start_routine_(arg_);
}

NsanThread::StackBounds NsanThread::GetStackBounds() const {
  if (!stack_switching_)
    return {stack_.bottom, stack_.top};
  const uptr cur_stack = GET_CURRENT_FRAME();
  // Note: need to check next stack first, because FinishSwitchFiber
  // may be in process of overwriting stack_.top/bottom_. But in such case
  // we are already on the next stack.
  if (cur_stack >= next_stack_.bottom && cur_stack < next_stack_.top)
    return {next_stack_.bottom, next_stack_.top};
  return {stack_.bottom, stack_.top};
}

uptr NsanThread::stack_top() { return GetStackBounds().top; }

uptr NsanThread::stack_bottom() { return GetStackBounds().bottom; }

bool NsanThread::AddrIsInStack(uptr addr) {
  const auto bounds = GetStackBounds();
  return addr >= bounds.bottom && addr < bounds.top;
}

void NsanThread::StartSwitchFiber(uptr bottom, uptr size) {
  CHECK(!stack_switching_);
  next_stack_.bottom = bottom;
  next_stack_.top = bottom + size;
  stack_switching_ = true;
}

void NsanThread::FinishSwitchFiber(uptr *bottom_old, uptr *size_old) {
  CHECK(stack_switching_);
  if (bottom_old)
    *bottom_old = stack_.bottom;
  if (size_old)
    *size_old = stack_.top - stack_.bottom;
  stack_.bottom = next_stack_.bottom;
  stack_.top = next_stack_.top;
  stack_switching_ = false;
  next_stack_.top = 0;
  next_stack_.bottom = 0;
}

static pthread_key_t tsd_key;
static bool tsd_key_inited;

void __nsan::NsanTSDInit(void (*destructor)(void *tsd)) {
  CHECK(!tsd_key_inited);
  tsd_key_inited = true;
  CHECK_EQ(0, pthread_key_create(&tsd_key, destructor));
}

static THREADLOCAL NsanThread *nsan_current_thread;

NsanThread *__nsan::GetCurrentThread() { return nsan_current_thread; }

void __nsan::SetCurrentThread(NsanThread *t) {
  // Make sure we do not reset the current NsanThread.
  CHECK_EQ(0, nsan_current_thread);
  nsan_current_thread = t;
  // Make sure that NsanTSDDtor gets called at the end.
  CHECK(tsd_key_inited);
  pthread_setspecific(tsd_key, t);
}

void __nsan::NsanTSDDtor(void *tsd) {
  NsanThread *t = (NsanThread *)tsd;
  if (t->destructor_iterations_ > 1) {
    t->destructor_iterations_--;
    CHECK_EQ(0, pthread_setspecific(tsd_key, tsd));
    return;
  }
  nsan_current_thread = nullptr;
  // Make sure that signal handler can not see a stale current thread pointer.
  atomic_signal_fence(memory_order_seq_cst);
  NsanThread::TSDDtor(tsd);
}
