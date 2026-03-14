// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: tvos, watchos
// XFAIL: ios && !iossim
#include "sanitizer_common/sanitizer_ucontext.h"
#include "test.h"

char stack[64 * 1024] __attribute__((aligned(16)));

ucontext_t uc, orig_uc[2];
void *fiber, *orig_fiber[2];

const unsigned N = 1000;

__attribute__((noinline))
void switch0() {
  __tsan_switch_to_fiber(orig_fiber[0], 0);
  swapcontext(&uc, &orig_uc[0]);
}

void func() {
  for (;;) {
    switch0();
    __tsan_switch_to_fiber(orig_fiber[1], 0);
    swapcontext(&uc, &orig_uc[1]);
  }
}

void *Thread(void *x) {
  orig_fiber[1] = __tsan_get_current_fiber();
  for (unsigned i = 0; i < N; i++) {
    barrier_wait(&barrier);
    __tsan_switch_to_fiber(fiber, 0);
    swapcontext(&orig_uc[1], &uc);
    barrier_wait(&barrier);
  }
  return 0;
}

int main() {
  fiber = __tsan_create_fiber(0);
  barrier_init(&barrier, 2);
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  orig_fiber[0] = __tsan_get_current_fiber();
  getcontext(&uc);
  uc.uc_stack.ss_sp = stack;
  uc.uc_stack.ss_size = sizeof(stack);
  uc.uc_link = 0;
  makecontext(&uc, func, 0);
  for (unsigned i = 0; i < N; i++) {
    __tsan_switch_to_fiber(fiber, 0);
    swapcontext(&orig_uc[0], &uc);
    barrier_wait(&barrier);
    barrier_wait(&barrier);
  }
  pthread_join(t, 0);
  __tsan_destroy_fiber(fiber);
  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: PASS
