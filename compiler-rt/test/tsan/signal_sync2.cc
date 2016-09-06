// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: darwin
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>

// Test synchronization in signal handled within IgnoreSync region.

extern "C" void AnnotateIgnoreSyncBegin(const char *f, int l);
extern "C" void AnnotateIgnoreSyncEnd(const char *f, int l);

const int kSignalCount = 500;

__thread int process_signals;
int signals_handled;
int done;
int ready[kSignalCount];
long long data[kSignalCount];

static void handler(int sig) {
  if (!__atomic_load_n(&process_signals, __ATOMIC_RELAXED))
    return;
  int pos = signals_handled++;
  if (pos >= kSignalCount)
    return;
  data[pos] = pos;
  __atomic_store_n(&ready[pos], 1, __ATOMIC_RELEASE);
}

static void* thr(void *p) {
  AnnotateIgnoreSyncBegin(__FILE__, __LINE__);
  __atomic_store_n(&process_signals, 1, __ATOMIC_RELAXED);
  while (!__atomic_load_n(&done, __ATOMIC_RELAXED)) {
  }
  AnnotateIgnoreSyncEnd(__FILE__, __LINE__);
  return 0;
}

int main() {
  struct sigaction act = {};
  act.sa_handler = handler;
  if (sigaction(SIGPROF, &act, 0)) {
    perror("sigaction");
    exit(1);
  }
  itimerval t;
  t.it_value.tv_sec = 0;
  t.it_value.tv_usec = 10;
  t.it_interval = t.it_value;
  if (setitimer(ITIMER_PROF, &t, 0)) {
    perror("setitimer");
    exit(1);
  }

  pthread_t th;
  pthread_create(&th, 0, thr, 0);
  for (int pos = 0; pos < kSignalCount; pos++) {
    while (__atomic_load_n(&ready[pos], __ATOMIC_ACQUIRE) == 0) {
    }
    if (data[pos] != pos) {
      printf("at pos %d, expect %d, got %lld\n", pos, pos, data[pos]);
      exit(1);
    }
  }
  __atomic_store_n(&done, 1, __ATOMIC_RELAXED);
  pthread_join(th, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: DONE
