// RUN: %clang_tsan %s -o %t
// RUN: %deflake %run %t | FileCheck %s
//
// XFAIL: {{.*linux.*}}

#include <pthread.h>
#include <setjmp.h>
#include <stdio.h>
#include <unistd.h>

int globalVar;

void foo(jmp_buf *buf, char willExit) {
  if (willExit)
    globalVar = 1;
  _longjmp(*buf, 1);
}

void func2() {
  jmp_buf buf;
  const int reps = 1000;
  for (int i = 0; i < reps; i++) {
    if (_setjmp(buf) == 0) {
      foo(&buf, i == reps - 1);
    }
  }
}

void *writeGlobal(void *ctx) {
  func2();
  return NULL;
}

void *readGlobal(void *ctx) {
  sleep(1);
  printf("globalVar: %d\n", globalVar);
  return NULL;
}

int main() {
  pthread_t t[2];
  pthread_create(&t[0], NULL, writeGlobal, NULL);
  pthread_create(&t[1], NULL, readGlobal, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK:   Read of size 4 at {{.*}} by thread
// CHECK:     #0 readGlobal
// CHECK:   Previous write of size 4 at {{.*}} by thread
// CHECK:     #0 foo
// CHECK:     #1 func2
// CHECK:     #2 writeGlobal
