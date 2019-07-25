// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: tvos, watchos
// XFAIL: ios && !iossim
#include "sanitizer_common/sanitizer_ucontext.h"
#include "test.h"
#include <setjmp.h>

char stack[64 * 1024] __attribute__((aligned(16)));

sigjmp_buf jmpbuf, orig_jmpbuf[2];
void *fiber, *orig_fiber[2];

const unsigned N = 1000;

__attribute__((noinline))
void switch0() {
  if (!sigsetjmp(jmpbuf, 0)) {
    __tsan_switch_to_fiber(orig_fiber[0], 0);
    siglongjmp(orig_jmpbuf[0], 1);
  }
}

void func() {
  if (!sigsetjmp(jmpbuf, 0)) {
    __tsan_switch_to_fiber(orig_fiber[0], 0);
    siglongjmp(orig_jmpbuf[0], 1);
  }
  for (;;) {
    switch0();
    if (!sigsetjmp(jmpbuf, 0)) {
      __tsan_switch_to_fiber(orig_fiber[1], 0);
      siglongjmp(orig_jmpbuf[1], 1);
    }
  }
}

void *Thread(void *x) {
  orig_fiber[1] = __tsan_get_current_fiber();
  for (unsigned i = 0; i < N; i++) {
    barrier_wait(&barrier);
    if (!sigsetjmp(orig_jmpbuf[1], 0)) {
      __tsan_switch_to_fiber(fiber, 0);
      siglongjmp(jmpbuf, 1);
    }
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
  ucontext_t uc, orig_uc;
  getcontext(&uc);
  uc.uc_stack.ss_sp = stack;
  uc.uc_stack.ss_size = sizeof(stack);
  uc.uc_link = 0;
  makecontext(&uc, func, 0);
  if (!sigsetjmp(orig_jmpbuf[0], 0)) {
    __tsan_switch_to_fiber(fiber, 0);
    swapcontext(&orig_uc, &uc);
  }
  for (unsigned i = 0; i < N; i++) {
    if (!sigsetjmp(orig_jmpbuf[0], 0)) {
      __tsan_switch_to_fiber(fiber, 0);
      siglongjmp(jmpbuf, 1);
    }
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
