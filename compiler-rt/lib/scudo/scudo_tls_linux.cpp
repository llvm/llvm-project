//===-- scudo_tls_linux.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Scudo thread local structure implementation for platforms supporting
/// thread_local.
///
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_LINUX && !SANITIZER_ANDROID

#include "scudo_tls.h"

#include <pthread.h>

namespace __scudo {

static pthread_once_t GlobalInitialized = PTHREAD_ONCE_INIT;
static pthread_key_t PThreadKey;

__attribute__((tls_model("initial-exec")))
THREADLOCAL ThreadState ScudoThreadState = ThreadNotInitialized;
__attribute__((tls_model("initial-exec")))
THREADLOCAL ScudoThreadContext ThreadLocalContext;

static void teardownThread(void *Ptr) {
  uptr I = reinterpret_cast<uptr>(Ptr);
  // The glibc POSIX thread-local-storage deallocation routine calls user
  // provided destructors in a loop of PTHREAD_DESTRUCTOR_ITERATIONS.
  // We want to be called last since other destructors might call free and the
  // like, so we wait until PTHREAD_DESTRUCTOR_ITERATIONS before draining the
  // quarantine and swallowing the cache.
  if (I > 1) {
    // If pthread_setspecific fails, we will go ahead with the teardown.
    if (LIKELY(pthread_setspecific(PThreadKey,
                                   reinterpret_cast<void *>(I - 1)) == 0))
      return;
  }
  ThreadLocalContext.commitBack();
  ScudoThreadState = ThreadTornDown;
}


static void initOnce() {
  CHECK_EQ(pthread_key_create(&PThreadKey, teardownThread), 0);
  initScudo();
}

void initThread() {
  CHECK_EQ(pthread_once(&GlobalInitialized, initOnce), 0);
  CHECK_EQ(pthread_setspecific(PThreadKey, reinterpret_cast<void *>(
      GetPthreadDestructorIterations())), 0);
  ThreadLocalContext.init();
  ScudoThreadState = ThreadInitialized;
}

}  // namespace __scudo

#endif  // SANITIZER_LINUX && !SANITIZER_ANDROID
