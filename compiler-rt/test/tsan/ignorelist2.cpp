// Test that ignorelisted functions are still contained in the stack trace.

// RUN: echo "fun:*Ignorelisted_Thread2*" > %t.ignorelist
// RUN: echo "fun:*CallTouchGlobal*" >> %t.ignorelist

// RUN: %clangxx_tsan -O1 %s -fsanitize-ignorelist=%t.ignorelist -o %t
// RUN: %deflake %run %t 2>&1 | FileCheck %s
#include "test.h"

int Global;

void *Thread1(void *x) {
  barrier_wait(&barrier);
  // CHECK: ThreadSanitizer: data race
  // CHECK: Write of size 4
  // CHECK: #0 Thread1{{.*}}ignorelist2.cpp:[[@LINE+1]]
  Global++;
  return NULL;
}

void TouchGlobal() __attribute__((noinline)) {
  // CHECK: Previous write of size 4
  // CHECK: #0 TouchGlobal{{.*}}ignorelist2.cpp:[[@LINE+1]]
  Global--;
}

void CallTouchGlobal() __attribute__((noinline)) {
  // CHECK: #1 CallTouchGlobal{{.*}}ignorelist2.cpp:[[@LINE+1]]
  TouchGlobal();
}

void *Ignorelisted_Thread2(void *x) {
  Global--;
  // CHECK: #2 Ignorelisted_Thread2{{.*}}ignorelist2.cpp:[[@LINE+1]]
  CallTouchGlobal();
  barrier_wait(&barrier);
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Ignorelisted_Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  fprintf(stderr, "PASS\n");
  return 0;
}
