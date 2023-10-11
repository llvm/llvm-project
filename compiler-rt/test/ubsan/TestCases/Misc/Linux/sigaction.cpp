// RUN: %clangxx -fsanitize=undefined -Wno-prio-ctor-dtor -shared-libsan %s -o %t && %run %t 2>&1 | FileCheck %s

// Ensure ubsan runtime/interceptors are lazily initialized if called early.

// The test seems to segfault on aarch64 with tsan:
// https://lab.llvm.org/buildbot/#/builders/179/builds/6662
// Reason unknown, needs debugging.
// UNSUPPORTED: target=aarch64{{.*}} && ubsan-tsan

#include <assert.h>
#include <signal.h>
#include <stdio.h>

__attribute__((constructor(1))) void ctor() {
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
