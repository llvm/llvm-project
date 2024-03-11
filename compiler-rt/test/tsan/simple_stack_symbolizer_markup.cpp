// REQUIRES: linux
// RUN: rm -rf %t
// RUN: mkdir -p %t/.build-id/12
// RUN: %clangxx_tsan %s -Wl,--build-id=0x12345678 -O1 -o %t/main
// RUN: cp %t/main %t/.build-id/12/345678.debug
// RUN: %env_tsan_opts=enable_symbolizer_markup=1 %deflake %run %t/main >%t/sanitizer.out
// RUN: llvm-symbolizer --filter-markup --debug-file-directory=%t < %t/sanitizer.out | FileCheck %s

#include "test.h"

int Global;

[[gnu::noinline]] void foo1() { Global = 42; }

[[gnu::noinline]] void bar1() {
  volatile int tmp = 42;
  int tmp2 = tmp;
  (void)tmp2;
  foo1();
}

[[gnu::noinline]] void foo2() {
  volatile int tmp = Global;
  int tmp2 = tmp;
  (void)tmp2;
}

[[gnu::noinline]] void bar2() {
  volatile int tmp = 42;
  int tmp2 = tmp;
  (void)tmp2;
  foo2();
}

void *Thread1(void *x) {
  barrier_wait(&barrier);
  bar1();
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, NULL, Thread1, NULL);
  bar2();
  barrier_wait(&barrier);
  pthread_join(t, NULL);
}

//      CHECK: WARNING: ThreadSanitizer: data race
//      CHECK:   Write of size 4 at {{0x.*}} by thread T1:
//      CHECK:     #0 {{0x.*}} foo1{{.*}}simple_stack_symbolizer_markup.cpp:[[#@LINE - 39]]
// CHECK-NEXT:     #1 {{0x.*}} bar1{{.*}}simple_stack_symbolizer_markup.cpp:[[#@LINE - 34]]
// CHECK-NEXT:     #2 {{0x.*}} Thread1{{.*}}simple_stack_symbolizer_markup.cpp:[[#@LINE - 17]]
//      CHECK:   Previous read of size 4 at {{.*}} by main thread:
// CHECK-NEXT:     #0 {{0x.*}} foo2{{.*}}simple_stack_symbolizer_markup.cpp:[[#@LINE - 33]]
// CHECK-NEXT:     #1 {{0x.*}} bar2{{.*}}simple_stack_symbolizer_markup.cpp:[[#@LINE - 25]]
// CHECK-NEXT:     #2 {{0x.*}} main{{.*}}simple_stack_symbolizer_markup.cpp:[[#@LINE - 13]]
