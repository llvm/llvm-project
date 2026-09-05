// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,deadcode.DeadStores -verify %s
// expected-no-diagnostics

// Regression test for https://github.com/llvm/llvm-project/issues/160527:
// a cleanup function that frees the pointee must not produce a false
// "Potential leak" at an explicit return, and the assignment must not be
// reported as a dead store.

#include "Inputs/system-header-simulator-for-malloc.h"

void *my_malloc(size_t size) {
  return malloc(size);
}

// The cleanup function. It takes a pointer to the variable, so char** for a
// char* variable (codegen calls free_pointer(&data)).
static void free_pointer(char **p) {
  free(*p);
}

void process_data(void) {
  // The variable 'data' is tied to the 'free_pointer' function.
  __attribute__((cleanup(free_pointer))) char *data = my_malloc(100);

  if (!data) {
    return;
  }

  // No explicit free(data) here: the cleanup function is called automatically
  // when process_data() returns.
  return;
} // no leak on any path, and no dead-store warning on 'data'.
