// RUN: %clangxx_tsan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
#include "test.h"

int Global;
volatile int x;
const int kSize = 64 << 10;
volatile long data[kSize];

void __attribute__((noinline)) foo() {
  for (int i = 0; i < kSize; i++)
    data[i]++;
}

void *Thread(void *a) {
  __atomic_store_n(&x, 1, __ATOMIC_RELEASE);
  foo();
  data[0]++;
  if (a != 0)
    barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  for (int i = 0; i < 50; i++) {
    pthread_t t;
    pthread_create(&t, 0, Thread, 0);
    pthread_join(t, 0);
  }
  pthread_t t;
  pthread_create(&t, 0, Thread, (void*)1);
  barrier_wait(&barrier);
  for (int i = 0; i < kSize; i++)
    data[i]++;
  pthread_join(t, 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// Previously this test produced bogus stack traces like:
//   Previous write of size 8 at 0x0000006a8ff8 by thread T17:
//     #0 foo() restore_stack.cpp:13:5 (restore_stack.cpp.exe+0x00000040622c)
//     #1 Thread(void*) restore_stack.cpp:18:3 (restore_stack.cpp.exe+0x000000406283)
//     #2 __tsan_thread_start_func rtl/tsan_interceptors.cpp:886 (restore_stack.cpp.exe+0x00000040a749)
//     #3 Thread(void*) restore_stack.cpp:18:3 (restore_stack.cpp.exe+0x000000406283)

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NOT: __tsan_thread_start_func
// CHECK-NOT: #3 Thread
// CHECK: DONE
