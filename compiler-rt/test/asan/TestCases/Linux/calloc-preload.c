// Test that initially callocked memory is properly freed
// (see https://github.com/google/sanitizers/issues/626).
// 
// RUN: %clang %s -o %t
// RUN: env LD_PRELOAD=%shared_libasan %run %t
//
// REQUIRES: asan-dynamic-runtime
//
// This way of setting LD_PRELOAD does not work with Android test runner.
// REQUIRES: !android

#include <stdio.h>
#include <stdlib.h>

static void *ptr;

// This constructor will run before __asan_init
// so calloc will allocate memory from special pool.
static void init() {
  ptr = calloc(10, 1);
}

__attribute__((section(".preinit_array"), used))
void *dummy = init;

void free_memory() {
  // This used to abort because
  // Asan's free didn't recognize ptr.
  free(ptr);
}

int main() {
  free_memory();
  return 0;
}

