// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: tvos, watchos
#include "sanitizer_common/sanitizer_ucontext.h"
#include "test.h"

char stack[64 * 1024] __attribute__((aligned(16)));

ucontext_t uc, orig_uc1, orig_uc2;
void *fiber, *orig_fiber1, *orig_fiber2;

int var;

void *Thread(void *x) {
  orig_fiber2 = __tsan_get_current_fiber();
  swapcontext(&orig_uc2, &orig_uc1);
  return 0;
}

void func() {
  pthread_t t;
  pthread_create(&t, 0, Thread, 0);
  pthread_join(t, 0);
  __tsan_switch_to_fiber(orig_fiber1, 0);
  swapcontext(&uc, &orig_uc1);
}

int main() {
  orig_fiber1 = __tsan_get_current_fiber();
  fiber = __tsan_create_fiber(0);
  getcontext(&uc);
  uc.uc_stack.ss_sp = stack;
  uc.uc_stack.ss_size = sizeof(stack);
  uc.uc_link = 0;
  makecontext(&uc, func, 0);
  var = 1;
  __tsan_switch_to_fiber(fiber, 0);
  swapcontext(&orig_uc1, &uc);
  var = 2;
  __tsan_switch_to_fiber(orig_fiber2, 0);
  swapcontext(&orig_uc1, &orig_uc2);
  var = 3;
  __tsan_destroy_fiber(fiber);
  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: PASS
