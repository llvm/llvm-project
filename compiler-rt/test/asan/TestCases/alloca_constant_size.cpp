// Regression test for https://github.com/google/sanitizers/issues/691

// RUN: %clangxx_asan -O0 %s -o %t -fstack-protector
// RUN: %run %t 1 2>&1 | FileCheck %s
// RUN: %run %t 2 2>&1 | FileCheck %s

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// MSVC provides _alloca instead of alloca.
#if defined(_MSC_VER) && !defined(alloca)
# define alloca _alloca
#endif

#if defined(__sun__) && defined(__svr4__)
#include <alloca.h>
#endif


void f1_alloca() {
  char *dynamic_buffer = (char *)alloca(200);
  fprintf(stderr, "dynamic_buffer = %p\n", dynamic_buffer);
  memset(dynamic_buffer, 'y', 200);
  return;
}

static const int kDynamicArraySize = 200;

void f1_vla() {
  char dynamic_buffer[kDynamicArraySize];
  fprintf(stderr, "dynamic_buffer = %p\n", dynamic_buffer);
  memset(dynamic_buffer, 'y', kDynamicArraySize);
  return;
}

void f2() {
  char buf[1024];
  memset(buf, 'x', 1024);
}

int main(int argc, const char *argv[]) {
  if (!strcmp(argv[1], "1")) {
    f1_alloca();
  } else if (!strcmp(argv[1], "2")) {
    f1_vla();
  }
  f2();
  fprintf(stderr, "Done.\n");
  return 0;
}

// CHECK-NOT: ERROR: AddressSanitizer
// CHECK: Done.
