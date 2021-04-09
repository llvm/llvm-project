// Ensure that operator new/delete are still replaceable.

// RUN: %clangxx %s -o %t -fsanitize=address -shared-libsan && not %run %t 2>&1 | FileCheck %s
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
