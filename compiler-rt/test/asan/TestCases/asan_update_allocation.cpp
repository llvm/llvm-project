// RUN: %clangxx_asan -O0 %s --std=c++11 -o %t

// RUN: not %run %t 10 0 2>&1 | FileCheck %s --check-prefixes=CHECK,T0
// RUN: not %run %t 10000000 0 2>&1 | FileCheck %s --check-prefixes=CHECK,T0

// RUN: not %run %t 10 1 2>&1 | FileCheck %s --check-prefixes=CHECK,T1
// RUN: not %run %t 10000000 1 2>&1 | FileCheck %s --check-prefixes=CHECK,T1

// REQUIRES: stable-runtime

#include <sanitizer/asan_interface.h>
#include <stdlib.h>
#include <thread>

void UPDATE(void *p) {
  __asan_update_allocation_context(p);
}

int main(int argc, char *argv[]) {
  char *x = (char *)malloc(atoi(argv[1]) * sizeof(char));
  if (atoi(argv[2]))
    std::thread([&]() { UPDATE(x); }).join();
  else
    UPDATE(x);
  free(x);
  return x[5];
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK: READ of size 1 at {{.*}} thread T0
  // T0: allocated by thread T0 here
  // T1: allocated by thread T1 here
  // CHECK: UPDATE
}
