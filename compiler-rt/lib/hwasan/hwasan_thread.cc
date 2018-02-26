
#include "hwasan.h"
#include "hwasan_thread.h"
#include "hwasan_poisoning.h"
#include "hwasan_interface_internal.h"

#include "sanitizer_common/sanitizer_tls_get_addr.h"

namespace __hwasan {

HwasanThread *HwasanThread::Create(thread_callback_t start_routine,
                               void *arg) {
  uptr PageSize = GetPageSizeCached();
  uptr size = RoundUpTo(sizeof(HwasanThread), PageSize);
  HwasanThread *thread = (HwasanThread*)MmapOrDie(size, __func__);
  thread->start_routine_ = start_routine;
  thread->arg_ = arg;
  thread->destructor_iterations_ = GetPthreadDestructorIterations();

  return thread;
}

void HwasanThread::SetThreadStackAndTls() {
  uptr tls_size = 0;
  uptr stack_size = 0;
  GetThreadStackAndTls(IsMainThread(), &stack_bottom_, &stack_size,
                       &tls_begin_, &tls_size);
  stack_top_ = stack_bottom_ + stack_size;
  tls_end_ = tls_begin_ + tls_size;

  int local;
  CHECK(AddrIsInStack((uptr)&local));
}

void HwasanThread::Init() {
  SetThreadStackAndTls();
  CHECK(MEM_IS_APP(stack_bottom_));
  CHECK(MEM_IS_APP(stack_top_ - 1));
}

void HwasanThread::TSDDtor(void *tsd) {
  HwasanThread *t = (HwasanThread*)tsd;
  t->Destroy();
}

void HwasanThread::ClearShadowForThreadStackAndTLS() {
  TagMemory(stack_bottom_, stack_top_ - stack_bottom_, 0);
  if (tls_begin_ != tls_end_)
    TagMemory(tls_begin_, tls_end_ - tls_begin_, 0);
}

void HwasanThread::Destroy() {
  malloc_storage().CommitBack();
  ClearShadowForThreadStackAndTLS();
  uptr size = RoundUpTo(sizeof(HwasanThread), GetPageSizeCached());
  UnmapOrDie(this, size);
  DTLS_Destroy();
}

thread_return_t HwasanThread::ThreadStart() {
  Init();

  if (!start_routine_) {
    // start_routine_ == 0 if we're on the main thread or on one of the
    // OS X libdispatch worker threads. But nobody is supposed to call
    // ThreadStart() for the worker threads.
    return 0;
  }

  thread_return_t res = start_routine_(arg_);

  return res;
}

} // namespace __hwasan
