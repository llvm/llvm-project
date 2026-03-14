// RUN: %clang_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>

void bar(jmp_buf env) {
  volatile int x = 42;
  longjmp(env, 42);
  x++;
}

void foo(jmp_buf env) {
  volatile int x = 42;
  bar(env);
  x++;
}

__attribute__((noinline)) void badguy() {
  pthread_mutex_t mtx;
  pthread_mutex_init(&mtx, 0);
  pthread_mutex_lock(&mtx);
  pthread_mutex_destroy(&mtx);
}

__attribute__((noinline)) void mymain() {
  jmp_buf env;
  if (setjmp(env) == 42) {
    badguy();
    return;
  }
  foo(env);
  fprintf(stderr, "FAILED\n");
}

int main() {
  volatile int x = 42;
  mymain();
  return x;
}

// CHECK-NOT: FAILED
// CHECK: WARNING: ThreadSanitizer: destroy of a locked mutex
// CHECK:   #0 pthread_mutex_destroy
// CHECK:   #1 badguy
// CHECK:   #2 mymain
// CHECK:   #3 main

