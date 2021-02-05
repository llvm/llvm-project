#include "dfsan_thread.h"

#include <pthread.h>

#include "dfsan.h"

namespace __dfsan {

DFsanThread *DFsanThread::Create(void *start_routine_trampoline,
                                 thread_callback_t start_routine, void *arg) {
  uptr PageSize = GetPageSizeCached();
  uptr size = RoundUpTo(sizeof(DFsanThread), PageSize);
  DFsanThread *thread = (DFsanThread *)MmapOrDie(size, __func__);
  thread->start_routine_trampoline_ = start_routine_trampoline;
  thread->start_routine_ = start_routine;
  thread->arg_ = arg;
  thread->destructor_iterations_ = GetPthreadDestructorIterations();

  return thread;
}

void DFsanThread::SetThreadStackAndTls() {
  uptr tls_size = 0;
  uptr stack_size = 0;
  uptr tls_begin;
  GetThreadStackAndTls(IsMainThread(), &stack_.bottom, &stack_size, &tls_begin,
                       &tls_size);
  stack_.top = stack_.bottom + stack_size;

  int local;
  CHECK(AddrIsInStack((uptr)&local));
}

void DFsanThread::Init() { SetThreadStackAndTls(); }

void DFsanThread::TSDDtor(void *tsd) {
  DFsanThread *t = (DFsanThread *)tsd;
  t->Destroy();
}

void DFsanThread::Destroy() {
  uptr size = RoundUpTo(sizeof(DFsanThread), GetPageSizeCached());
  UnmapOrDie(this, size);
}

thread_return_t DFsanThread::ThreadStart() {
  Init();

  if (!start_routine_) {
    // start_routine_ == 0 if we're on the main thread or on one of the
    // OS X libdispatch worker threads. But nobody is supposed to call
    // ThreadStart() for the worker threads.
    return 0;
  }

  CHECK(start_routine_trampoline_);

  typedef void *(*thread_callback_trampoline_t)(void *, void *, dfsan_label,
                                                dfsan_label *);

  dfsan_label ret_label;
  return ((thread_callback_trampoline_t)
              start_routine_trampoline_)((void *)start_routine_, arg_, 0,
                                         &ret_label);
}

DFsanThread::StackBounds DFsanThread::GetStackBounds() const {
  return {stack_.bottom, stack_.top};
}

uptr DFsanThread::stack_top() { return GetStackBounds().top; }

uptr DFsanThread::stack_bottom() { return GetStackBounds().bottom; }

bool DFsanThread::AddrIsInStack(uptr addr) {
  const auto bounds = GetStackBounds();
  return addr >= bounds.bottom && addr < bounds.top;
}

static pthread_key_t tsd_key;
static bool tsd_key_inited = false;

void DFsanTSDInit(void (*destructor)(void *tsd)) {
  CHECK(!tsd_key_inited);
  tsd_key_inited = true;
  CHECK_EQ(0, pthread_key_create(&tsd_key, destructor));
}

static THREADLOCAL DFsanThread *dfsan_current_thread;

DFsanThread *GetCurrentThread() { return dfsan_current_thread; }

void SetCurrentThread(DFsanThread *t) {
  // Make sure we do not reset the current DFsanThread.
  CHECK_EQ(0, dfsan_current_thread);
  dfsan_current_thread = t;
  // Make sure that DFsanTSDDtor gets called at the end.
  CHECK(tsd_key_inited);
  pthread_setspecific(tsd_key, t);
}

void DFsanTSDDtor(void *tsd) {
  DFsanThread *t = (DFsanThread *)tsd;
  if (t->destructor_iterations_ > 1) {
    t->destructor_iterations_--;
    CHECK_EQ(0, pthread_setspecific(tsd_key, tsd));
    return;
  }
  dfsan_current_thread = nullptr;
  // Make sure that signal handler can not see a stale current thread pointer.
  atomic_signal_fence(memory_order_seq_cst);
  DFsanThread::TSDDtor(tsd);
}

}  // namespace __dfsan
