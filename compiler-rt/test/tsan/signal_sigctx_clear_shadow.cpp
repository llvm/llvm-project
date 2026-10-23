// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: darwin

#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>

#include "test.h"

extern "C" {
void __tsan_write4(void *addr);
}

static volatile int g_sig_ready = 0;
static volatile int g_sig_handled = 0;
static void *volatile g_siginfo_code_addr = nullptr;

static void sigusr1_handler(int sig, siginfo_t *info, void *uctx) {
  // Instrumented read of info->si_code inside ThreadSignalContext.
  volatile int code = info->si_code;
  (void)code;
  AnnotateIgnoreWritesBegin(__FILE__, __LINE__);
  g_siginfo_code_addr = &info->si_code;
  g_sig_handled = 1;
  AnnotateIgnoreWritesEnd(__FILE__, __LINE__);
}

static void *signal_worker(void *arg) {
  AnnotateIgnoreWritesBegin(__FILE__, __LINE__);
  AnnotateIgnoreSyncBegin(__FILE__, __LINE__);
  __atomic_store_n(&g_sig_ready, 1, __ATOMIC_RELAXED);
  AnnotateIgnoreSyncEnd(__FILE__, __LINE__);
  AnnotateIgnoreWritesEnd(__FILE__, __LINE__);

  for (;;) {
    AnnotateIgnoreReadsBegin(__FILE__, __LINE__);
    int handled = g_sig_handled;
    AnnotateIgnoreReadsEnd(__FILE__, __LINE__);
    if (handled)
      break;
    // pthread_sigmask is intercepted by TSan and processes pending signals on
    // return.
    pthread_sigmask(SIG_BLOCK, nullptr, nullptr);
  }
  return nullptr;
}

int main() {
  struct sigaction sa = {};
  sa.sa_sigaction = sigusr1_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO;
  assert(sigaction(SIGUSR1, &sa, nullptr) == 0);

  pthread_t th;
  pthread_create(&th, nullptr, signal_worker, nullptr);

  for (;;) {
    AnnotateIgnoreReadsBegin(__FILE__, __LINE__);
    AnnotateIgnoreSyncBegin(__FILE__, __LINE__);
    int ready = __atomic_load_n(&g_sig_ready, __ATOMIC_RELAXED);
    AnnotateIgnoreSyncEnd(__FILE__, __LINE__);
    AnnotateIgnoreReadsEnd(__FILE__, __LINE__);
    if (ready)
      break;
    sched_yield();
  }

  assert(pthread_kill(th, SIGUSR1) == 0);

  // Join with sync ignored so the main thread does not acquire the worker
  // thread's vector clock; only PlatformCleanUpThreadState clearing
  // ThreadSignalContext shadow prevents a false-positive race on
  // g_siginfo_code_addr.
  AnnotateIgnoreSyncBegin(__FILE__, __LINE__);
  pthread_join(th, nullptr);
  AnnotateIgnoreSyncEnd(__FILE__, __LINE__);

  AnnotateIgnoreReadsBegin(__FILE__, __LINE__);
  void *addr = (void *)g_siginfo_code_addr;
  AnnotateIgnoreReadsEnd(__FILE__, __LINE__);
  assert(addr != nullptr);

  __tsan_write4(addr);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
