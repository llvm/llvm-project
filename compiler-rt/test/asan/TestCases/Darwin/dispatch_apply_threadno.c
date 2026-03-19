// Bugs caught within missing GCD dispatch blocks result in thread being reported as T-1
// with an empty stack.
// This tests that dispatch_apply blocks can capture valid thread number and stack.

// RUN: %clang_asan %s -o %t
// RUN: not %run %t func 2>&1 | FileCheck %s --check-prefixes=CHECK-FUNC,CHECK
// RUN: not %run %t block 2>&1 | FileCheck %s --check-prefixes=CHECK-BLOCK,CHECK

#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((noinline)) void access_memory_frame(char *x) { *x = 0; }

__attribute__((noinline)) void test_dispatch_apply() {
  char *x = (char *)malloc(4);
  dispatch_apply(8, dispatch_get_global_queue(0, 0), ^(size_t i) {
    access_memory_frame(&x[i]);
  });
}

typedef struct {
  char *data;
} Context;

void da_func(void *ctx, size_t i) {
  Context *c = (Context *)ctx;
  access_memory_frame(&c->data[i]);
}

__attribute__((noinline)) void test_dispatch_apply_f() {
  Context *ctx = (Context *)malloc(sizeof(Context));
  ctx->data = (char *)malloc(4);
  dispatch_apply_f(8, dispatch_get_global_queue(0, 0), ctx, da_func);
}

int main(int argc, const char *argv[]) {
  if (strcmp(argv[1], "func") == 0) {
    fprintf(stderr, "Test dispatch_apply with function\n");
    // CHECK-FUNC: dispatch_apply with function
    test_dispatch_apply_f();
  } else if (strcmp(argv[1], "block") == 0) {
    fprintf(stderr, "Test dispatch_apply with block\n");
    // CHECK-BLOCK: dispatch_apply with block
    test_dispatch_apply();
  } else {
    abort();
  }
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK: #0 0x{{.*}} in {{.*}}access_memory_frame
// CHECK-NOT: T-1
