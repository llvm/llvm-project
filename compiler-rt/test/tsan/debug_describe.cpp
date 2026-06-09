// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %deflake %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

extern "C" {
void __tsan_on_report(void *report);
int __tsan_get_report_data(void *report, const char **description, int *count,
                           int *stack_count, int *mop_count, int *loc_count,
                           int *mutex_count, int *thread_count,
                           int *unique_tid_count, void **sleep_trace,
                           unsigned long trace_size);
int __tsan_describe_mop(void *report, unsigned long idx, char *out,
                        unsigned long outlen);
int __tsan_describe_loc(void *report, unsigned long idx, char *out,
                        unsigned long outlen);
int __tsan_describe_thread(void *report, unsigned long idx, char *out,
                           unsigned long outlen);
}

long my_global;

void *Thread(void *a) {
  barrier_wait(&barrier);
  my_global = 42;
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  my_global = 41;
  barrier_wait(&barrier);
  pthread_join(t, 0);
  fprintf(stderr, "Done.\n");
}

// Required for dyld macOS 12.0+
#if (__APPLE__)
__attribute__((weak))
#endif
__attribute__((disable_sanitizer_instrumentation)) extern "C" void
__tsan_on_report(void *report) {
  const char *description;
  int count, stack_count, mop_count, loc_count, mutex_count, thread_count,
      unique_tid_count;
  void *sleep_trace[16] = {0};
  __tsan_get_report_data(report, &description, &count, &stack_count, &mop_count,
                         &loc_count, &mutex_count, &thread_count,
                         &unique_tid_count, sleep_trace, 16);

  char buf[256];

  // Two mops: idx 0 is described as the primary access ("Write"); idx 1 is
  // described as the secondary access ("Previous write"). Output starts with
  // two spaces of indent.
  __tsan_describe_mop(report, 0, buf, sizeof(buf));
  fprintf(stderr, "mop0: [%s]\n", buf);
  // CHECK: mop0: [{{ +}}Write of size 8 at 0x{{[0-9a-f]+}} by thread T1]

  __tsan_describe_mop(report, 1, buf, sizeof(buf));
  fprintf(stderr, "mop1: [%s]\n", buf);
  // CHECK: mop1: [{{ +}}Previous write of size 8 at 0x{{[0-9a-f]+}} by main thread]

  // Location is a global; describe_loc returns 0 (no stack follows).
  if (loc_count > 0) {
    int has_stack = __tsan_describe_loc(report, 0, buf, sizeof(buf));
    fprintf(stderr, "loc0 has_stack=%d\n", has_stack);
    fprintf(stderr, "loc0: [%s]\n", buf);
    // CHECK: loc0 has_stack=0
    // CHECK: loc0: [{{.*}}global{{.*}}my_global{{.*}}]
  }

  // thread[0] = spawned thread, has_stack=1; thread[1] = main, has_stack=0.
  int t0_has_stack = __tsan_describe_thread(report, 0, buf, sizeof(buf));
  fprintf(stderr, "thread0 has_stack=%d\n", t0_has_stack);
  fprintf(stderr, "thread0: [%s]\n", buf);
  // CHECK: thread0 has_stack=1
  // CHECK: thread0: [{{.*}}Thread T1{{.*}}created by main thread{{.*}}at:]

  int t1_has_stack = __tsan_describe_thread(report, 1, buf, sizeof(buf));
  fprintf(stderr, "thread1 has_stack=%d\n", t1_has_stack);
  // The main thread is described but no stack follows (parent is itself
  // kInvalidTid in TSan; the function still returns 1 since it isn't a
  // GCD worker). We pin only that the description was written.
  fprintf(stderr, "thread1: [%s]\n", buf);
  // CHECK: thread1 has_stack={{[01]}}
  // CHECK: thread1: [{{.*}}Thread T0{{.*}}]

  // Truncation: outlen = 8 leaves room for 7 chars of description + null.
  // Description starts with "  Write" so we expect strlen == 7.
  char tiny[8];
  memset(tiny, 'X', sizeof(tiny));
  __tsan_describe_mop(report, 0, tiny, sizeof(tiny));
  fprintf(stderr, "tiny strlen=%zu\n", strlen(tiny));
  // CHECK: tiny strlen=7

  // outlen = 0 must not write anything (sentinel preserved).
  char sentinel[4] = {'A', 'B', 'C', 'D'};
  __tsan_describe_mop(report, 0, sentinel, 0);
  fprintf(stderr, "sentinel: %c%c%c%c\n", sentinel[0], sentinel[1], sentinel[2],
          sentinel[3]);
  // CHECK: sentinel: ABCD
}

// CHECK: Done.
// CHECK: ThreadSanitizer: reported 1 warnings
