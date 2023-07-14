// RUN: %clangxx -fsanitize=undefined -shared-libsan %s -o %t && %run %t 2>&1 | FileCheck %s

// Ensure ubsan runtime/interceptors are lazily initialized if called early.

#include <assert.h>
#include <signal.h>
#include <stdio.h>

__attribute__((constructor(0))) void ctor() {
  fprintf(stderr, "INIT\n");
  struct sigaction old;
  assert(!sigaction(SIGSEGV, nullptr, &old));
};

int main() {
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: INIT
// CHECK: DONE
