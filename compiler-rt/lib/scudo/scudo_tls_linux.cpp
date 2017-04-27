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

#if SANITIZER_LINUX

#include "scudo_tls.h"

#include <limits.h>
#include <pthread.h>

namespace __scudo {

static pthread_once_t GlobalInitialized = PTHREAD_ONCE_INIT;
static pthread_key_t PThreadKey;

thread_local ThreadState ScudoThreadState = ThreadNotInitialized;
thread_local ScudoThreadContext ThreadLocalContext;

static void teardownThread(void *Ptr) {
  uptr Iteration = reinterpret_cast<uptr>(Ptr);
  // The glibc POSIX thread-local-storage deallocation routine calls user
  // provided destructors in a loop of PTHREAD_DESTRUCTOR_ITERATIONS.
  // We want to be called last since other destructors might call free and the
  // like, so we wait until PTHREAD_DESTRUCTOR_ITERATIONS before draining the
  // quarantine and swallowing the cache.
  if (Iteration < PTHREAD_DESTRUCTOR_ITERATIONS) {
    pthread_setspecific(PThreadKey, reinterpret_cast<void *>(Iteration + 1));
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
  pthread_once(&GlobalInitialized, initOnce);
  pthread_setspecific(PThreadKey, reinterpret_cast<void *>(1));
  ThreadLocalContext.init();
  ScudoThreadState = ThreadInitialized;
}

}  // namespace __scudo

#endif  // SANITIZER_LINUX
