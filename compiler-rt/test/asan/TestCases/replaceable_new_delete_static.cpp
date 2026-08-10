// Ensure that operator new/delete are still replaceable using static-libsan.

// FIXME: Weak symbols aren't supported on Windows, although some code in
// compiler-rt already exists to solve this problem. We should probably define
// the new/delete interceptors as "weak" using those workarounds as well.
// UNSUPPORTED: target={{.*windows.*}}

// darwin only supports `shared-libsan`.
// REQUIRES: asan-static-runtime

// RUN: %clangxx %s -o %t -fsanitize=address -static-libsan && not %run %t 2>&1 | FileCheck %s

#include <cstdio>
#include <cstdlib>
#include <new>

void *operator new[](size_t size) {
  fprintf(stderr, "replaced new\n");
  return malloc(size);
}

void operator delete[](void *ptr) noexcept {
  fprintf(stderr, "replaced delete\n");
  return free(ptr);
}

int main(int argc, char **argv) {
  // CHECK: replaced new
  char *x = new char[5];
  // CHECK: replaced delete
  delete[] x;
  // CHECK: ERROR: AddressSanitizer
  *x = 13;
  return 0;
}
