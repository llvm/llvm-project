// Tests __hwasan_print_memory_usage.
// REQUIRES: shell
// RUN: %clang_hwasan %s -o %t
// RUN: ulimit -s 1000
// RUN: %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sanitizer/hwasan_interface.h>

int state;
__thread volatile char *sink;

__attribute__((noinline))
void *malloc_and_use(int size) {
  char *x = (char*)malloc(size);
  for (int i = 0; i < size; i++)
    x[i] = 42;  // make this memory used.
  return x;
}

void *T1(void *arg) {
  for (int i = 1; i <= (1 << 20); i *= 2) {
    sink = malloc_and_use(i);
    free(sink);
  }

  __sync_fetch_and_add(&state, 1);
  while (__sync_fetch_and_add(&state, 0) != 4) {}
  return NULL;
}

void *T4(void *arg) { return NULL; }

int main() {
  __hwasan_enable_allocator_tagging();
  sink = malloc_and_use(10);

  __hwasan_print_memory_usage();
  // CHECK: HWASAN pid: [[PID:[0-9]*]] rss: {{.*}} threads: 1 stacks: [[STACKS:[0-9]*]] thr_aux: {{.*}} stack_depot: {{.*}} uniq_stacks: [[UNIQ_STACKS:[0-9]*]] heap: [[HEAP:[0-9]*]]

  void *one_meg = malloc_and_use(1 << 20);

  __hwasan_print_memory_usage();
  // CHECK: HWASAN pid: [[PID]] rss: {{.*}} threads: 1 stacks: [[STACKS]] thr_aux: {{.*}} stack_depot: {{.*}}

  free(one_meg);

  __hwasan_print_memory_usage();
  // CHECK: HWASAN pid: [[PID]] rss: {{.*}} threads: 1 stacks: [[STACKS]] thr_aux: {{.*}} stack_depot: {{.*}} uniq_stacks: {{.*}} heap: [[HEAP]]

  pthread_t t1, t2, t3, t4;

  pthread_create(&t1, NULL, T1, NULL);
  pthread_create(&t2, NULL, T1, NULL);
  pthread_create(&t3, NULL, T1, NULL);
  pthread_create(&t4, NULL, T4, NULL);
  while (__sync_fetch_and_add(&state, 0) != 3) {}
  pthread_join(t4, NULL);

  __hwasan_print_memory_usage();
  // CHECK: HWASAN pid: [[PID]] rss: {{.*}} threads: 4 stacks:

  __sync_fetch_and_add(&state, 1);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  pthread_join(t3, NULL);
  __hwasan_print_memory_usage();
  // CHECK: HWASAN pid: [[PID]] rss: {{.*}} threads: 1 stacks: [[STACKS]]
}
