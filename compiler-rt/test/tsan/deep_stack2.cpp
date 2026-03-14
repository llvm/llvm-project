// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t | FileCheck %s
#include "test.h"

volatile long X;
volatile long Y;
volatile int N1 = 2 << 10;
volatile int N2 = 32 << 10;
void (*volatile F)();
void (*volatile G)();

static void foo() {
  if (--N1)
    return F();
  while (--N2)
    G();
}

static void bar() { Y++; }

void *Thread(void *p) {
  F();
  X = 43;
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  F = foo;
  G = bar;
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  barrier_wait(&barrier);
  X = 43;
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Write
// CHECK:     #0 main
// CHECK:   Previous write
// CHECK:     #0 Thread
