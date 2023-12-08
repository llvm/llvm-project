// RUN: %clangxx_tsan -O1 %s -o %t -DORDER1 && %deflake %run %t | FileCheck %s
// RUN: %clangxx_tsan -O1 %s -o %t -DORDER2 && %deflake %run %t | FileCheck %s
#include "test.h"

volatile int X;
volatile int N;
void (*volatile F)();

static void foo() {
  if (--N == 0)
    X = 42;
  else
    F();
}

void *Thread(void *p) {
#ifdef ORDER1
  barrier_wait(&barrier);
#endif
  F();
#ifdef ORDER2
  barrier_wait(&barrier);
#endif
  return 0;
}

static size_t RoundUp(size_t n, size_t to) {
  return ((n + to - 1) / to) * to;
}

int main() {
  barrier_init(&barrier, 2);
  N = 50000;
  F = foo;
  pthread_t t;
  pthread_attr_t a;
  pthread_attr_init(&a);
  size_t stack_size = N * 256 + (1 << 20);
  stack_size = RoundUp(stack_size, 0x10000);  // round the stack size to 64k
  int ret = pthread_attr_setstacksize(&a, stack_size);
  if (ret) abort();
  pthread_create(&t, &a, Thread, 0);
#ifdef ORDER2
  barrier_wait(&barrier);
#endif
  X = 43;
#ifdef ORDER1
  barrier_wait(&barrier);
#endif

  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:    #100 foo
// We must output sufficiently large stack (at least 100 frames)

