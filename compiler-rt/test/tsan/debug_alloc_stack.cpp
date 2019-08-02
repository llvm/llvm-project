// RUN: %clangxx_tsan -O0 %s -o %t
// RUN: env %env_tsan_opts=stack_trace_format=DEFAULT %deflake %run %t 2>&1 | FileCheck %s

// Until I figure out how to make this test work on Linux
// REQUIRES: system-darwin

#include "test.h"
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef __APPLE__
#include <sys/types.h>
#endif

extern "C" int __tsan_get_alloc_stack(void *addr, void **trace, size_t size,
                                      int *thread_id, uint64_t *os_id);

char *mem;
void alloc_func() { mem = (char *)malloc(10); }

void *AllocThread(void *context) {
  uint64_t tid;
#ifdef __APPLE__
  pthread_threadid_np(NULL, &tid);
#else
  tid = gettid();
#endif
  fprintf(stderr, "alloc stack thread os id = 0x%llx\n", tid);
  // CHECK: alloc stack thread os id = [[THREAD_OS_ID:0x[0-9a-f]+]]
  alloc_func();
  return NULL;
}

void *RaceThread(void *context) {
  *mem = 'a';
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  pthread_t t;
  barrier_init(&barrier, 2);

  pthread_create(&t, NULL, AllocThread, NULL);
  pthread_join(t, NULL);

  void *trace[100];
  size_t num_frames = 100;
  int thread_id;
  uint64_t *thread_os_id;
  num_frames =
      __tsan_get_alloc_stack(mem, trace, num_frames, &thread_id, &thread_os_id);

  fprintf(stderr, "alloc stack retval %s\n",
          (num_frames > 0 && num_frames < 10) ? "ok" : "");
  // CHECK: alloc stack retval ok
  fprintf(stderr, "thread id = %d\n", thread_id);
  // CHECK: thread id = 1
  fprintf(stderr, "thread os id = 0x%llx\n", thread_os_id);
  // CHECK: thread os id = [[THREAD_OS_ID]]
  fprintf(stderr, "%p\n", trace[0]);
  // CHECK: [[ALLOC_FRAME_0:0x[0-9a-f]+]]
  fprintf(stderr, "%p\n", trace[1]);
  // CHECK: [[ALLOC_FRAME_1:0x[0-9a-f]+]]
  fprintf(stderr, "%p\n", trace[2]);
  // CHECK: [[ALLOC_FRAME_2:0x[0-9a-f]+]]

  pthread_create(&t, NULL, RaceThread, NULL);
  barrier_wait(&barrier);
  mem[0] = 'b';
  pthread_join(t, NULL);

  free(mem);

  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK: Location is heap block of size 10 at {{.*}} allocated by thread T1
// CHECK: #0 [[ALLOC_FRAME_0]]
// CHECK: #1 [[ALLOC_FRAME_1]] in alloc_func
// CHECK: #2 [[ALLOC_FRAME_2]] in AllocThread
