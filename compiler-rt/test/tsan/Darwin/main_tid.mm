// RUN: %clang_tsan %s -o %t -framework Foundation
// RUN: %deflake %run %t 2>&1 | FileCheck %s

#import <pthread.h>
#import <stdio.h>
#import <stdlib.h>

extern "C" {
void __tsan_on_report(void *report);
int __tsan_get_report_thread(void *report, unsigned long idx, int *tid,
                             uint64_t *os_id, int *running,
                             const char **name, int *parent_tid, void **trace,
                             unsigned long trace_size);
}

void __tsan_on_report(void *report) {
  fprintf(stderr, "__tsan_on_report(%p)\n", report);

  int tid;
  uint64_t os_id;
  int running;
  const char *name;
  int parent_tid;
  void *trace[16] = {0};
  __tsan_get_report_thread(report, 0, &tid, &os_id, &running, &name, &parent_tid, trace, 16);
  fprintf(stderr, "tid = %d, os_id = %lu\n", tid, os_id);
}

int main() {
  fprintf(stderr, "Hello world.\n");

  uint64_t threadid;
  pthread_threadid_np(NULL, &threadid);
  fprintf(stderr, "pthread_threadid_np = %llu\n", threadid);

  pthread_mutex_t m;
  pthread_mutex_init(&m, NULL);
  pthread_mutex_unlock(&m);
  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: pthread_threadid_np = [[ADDR:[0-9]+]]
// CHECK: WARNING: ThreadSanitizer
// CHECK: tid = 0, os_id = [[ADDR]]
// CHECK: Done.
