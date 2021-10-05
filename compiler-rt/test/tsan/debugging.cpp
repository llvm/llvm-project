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
void *__tsan_get_current_report();
int __tsan_get_report_data(void *report, const char **description, int *count,
                           int *stack_count, int *mop_count, int *loc_count,
                           int *mutex_count, int *thread_count,
                           int *unique_tid_count, void **sleep_trace,
                           unsigned long trace_size);
int __tsan_get_report_mop(void *report, unsigned long idx, int *tid,
                          void **addr, int *size, int *write, int *atomic,
                          void **trace, unsigned long trace_size);
int __tsan_get_report_thread(void *report, unsigned long idx, int *tid,
                             uint64_t *os_id, int *running,
                             const char **name, int *parent_tid, void **trace,
                             unsigned long trace_size);
}

long my_global;

void *Thread(void *a) {
  barrier_wait(&barrier);
  my_global = 42;
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  fprintf(stderr, "&my_global = %p\n", &my_global);
  // CHECK: &my_global = [[GLOBAL:0x[0-9a-f]+]]
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  my_global = 41;
  barrier_wait(&barrier);
  pthread_join(t, 0);
  fprintf(stderr, "Done.\n");
}

__attribute__((disable_sanitizer_instrumentation)) void
__tsan_on_report(void *report) {
  fprintf(stderr, "__tsan_on_report(%p)\n", report);
  fprintf(stderr, "__tsan_get_current_report() = %p\n",
          __tsan_get_current_report());
  // CHECK: __tsan_on_report([[REPORT:0x[0-9a-f]+]])
  // CHECK: __tsan_get_current_report() = [[REPORT]]

  const char *description;
  int count;
  int stack_count, mop_count, loc_count, mutex_count, thread_count,
      unique_tid_count;
  void *sleep_trace[16] = {0};
  __tsan_get_report_data(report, &description, &count, &stack_count, &mop_count,
                         &loc_count, &mutex_count, &thread_count,
                         &unique_tid_count, sleep_trace, 16);
  fprintf(stderr, "report type = '%s', count = %d\n", description, count);
  // CHECK: report type = 'data-race', count = 0

  fprintf(stderr, "mop_count = %d\n", mop_count);
  // CHECK: mop_count = 2

  int tid;
  void *addr;
  int size, write, atomic;
  void *trace[16] = {0};

  __tsan_get_report_mop(report, 0, &tid, &addr, &size, &write, &atomic, trace,
                        16);
  fprintf(stderr, "tid = %d, addr = %p, size = %d, write = %d, atomic = %d\n",
          tid, addr, size, write, atomic);
  // CHECK: tid = 1, addr = [[GLOBAL]], size = 8, write = 1, atomic = 0
  fprintf(stderr, "trace[0] = %p, trace[1] = %p\n", trace[0], trace[1]);
  // CHECK: trace[0] = 0x{{[0-9a-f]+}}, trace[1] = {{0x0|\(nil\)|\(null\)}}

  __tsan_get_report_mop(report, 1, &tid, &addr, &size, &write, &atomic, trace,
                        16);
  fprintf(stderr, "tid = %d, addr = %p, size = %d, write = %d, atomic = %d\n",
          tid, addr, size, write, atomic);
  // CHECK: tid = 0, addr = [[GLOBAL]], size = 8, write = 1, atomic = 0
  fprintf(stderr, "trace[0] = %p, trace[1] = %p\n", trace[0], trace[1]);
  // CHECK: trace[0] = 0x{{[0-9a-f]+}}, trace[1] = {{0x0|\(nil\)|\(null\)}}

  fprintf(stderr, "thread_count = %d\n", thread_count);
  // CHECK: thread_count = 2

  uint64_t os_id;
  int running;
  const char *name;
  int parent_tid;

  __tsan_get_report_thread(report, 0, &tid, &os_id, &running, &name, &parent_tid, trace, 16);
  fprintf(stderr, "tid = %d\n", tid);
  // CHECK: tid = 1

  __tsan_get_report_thread(report, 1, &tid, &os_id, &running, &name, &parent_tid, trace, 16);
  fprintf(stderr, "tid = %d\n", tid);
  // CHECK: tid = 0
}

// CHECK: Done.
// CHECK: ThreadSanitizer: reported 1 warnings
