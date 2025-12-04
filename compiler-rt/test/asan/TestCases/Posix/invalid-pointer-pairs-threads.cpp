// On AIX, for 32 bit, the stack of main thread contains all other thread's stack.
// So we should be able to check invalid pointer based on the main thread stack, because
// all the stack address are in main thread's stack.
// However this is not true for 64 bit, for 64 bit, main thread stack does not overlap with
// other thread stack. This is same with other targets.
// See GetStackVariableShadowStart() for details.

// RUN: %clangxx_asan -O0 %s -pthread -o %t -mllvm -asan-detect-invalid-pointer-pair

// RUN: %if target={{.*aix.*}} && asan-32-bits %{ %env_asan_opts=detect_invalid_pointer_pairs=2 not %run %t a 2>&1 | FileCheck %s -check-prefix=AIX %} %else \
// RUN:   %{ %env_asan_opts=detect_invalid_pointer_pairs=2 %run %t a 2>&1 | FileCheck %s -check-prefix=OK -allow-empty %}
// RUN: %env_asan_opts=detect_invalid_pointer_pairs=2 not %run %t b 2>&1 | FileCheck %s -check-prefix=B

// pthread barriers are not available on OS X
// UNSUPPORTED: darwin

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

char *pointers[2];
pthread_barrier_t bar;

void *thread_main(void *n) {
  char local;

  unsigned long id = (unsigned long)n;
  pointers[id] = &local;
  pthread_barrier_wait(&bar);
  pthread_barrier_wait(&bar);

  return NULL;
}

int main(int argc, char **argv) {
  assert(argc >= 2);

  char t = argv[1][0];

  pthread_t threads[2];
  pthread_barrier_init(&bar, NULL, 3);
  pthread_create(&threads[0], 0, thread_main, (void *)0);
  pthread_create(&threads[1], 0, thread_main, (void *)1);
  pthread_barrier_wait(&bar);

  if (t == 'a') {
    // OK-NOT: not handled yet
    // AIX: ERROR: AddressSanitizer: invalid-pointer-pair
    // AIX: #{{[0-9]+ .*}} in .main {{.*}}invalid-pointer-pairs-threads.cpp:[[@LINE+1]]
    unsigned r = pointers[0] - pointers[1];
  } else {
    char local;
    char *parent_pointer = &local;

    // B: ERROR: AddressSanitizer: invalid-pointer-pair
    // B: #{{[0-9]+ .*}} in {{\.?main}} {{.*}}invalid-pointer-pairs-threads.cpp:[[@LINE+1]]
    unsigned r = parent_pointer - pointers[0];
  }

  pthread_barrier_wait(&bar);
  pthread_join(threads[0], 0);
  pthread_join(threads[1], 0);
  pthread_barrier_destroy(&bar);

  return 0;
}
