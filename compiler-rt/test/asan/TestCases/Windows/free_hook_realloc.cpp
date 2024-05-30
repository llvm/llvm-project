// Check that free hook doesn't conflict with Realloc.
// RUN: %clangxx_asan -O2 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// FIXME: merge this with the common free_hook_realloc test when we can run
// common tests on Windows.

#include <stdlib.h>
#include <io.h>
#include <sanitizer/allocator_interface.h>

static void *glob_ptr;

extern "C" {
void __sanitizer_free_hook(const volatile void *ptr) {
  if (ptr == glob_ptr) {
    *(int*)ptr = 0;
    write(1, "FreeHook\n", sizeof("FreeHook\n"));
  }
}
}

int main() {
  int *x = (int*)malloc(100);
  x[0] = 42;
  glob_ptr = x;
  int *y = (int*)realloc(x, 200);
  // Verify that free hook was called and didn't spoil the memory.
  if (y[0] != 42) {
    _exit(1);
  }
  write(1, "Passed\n", sizeof("Passed\n"));
  free(y);
  // CHECK: FreeHook
  // CHECK: Passed
  return 0;
}
