// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <stdio.h>
#include <stdlib.h>

// Defined by tsan.
extern "C" FILE *__interceptor_fopen(const char *file, const char *mode);
extern "C" int __interceptor_fileno(FILE *f);

extern "C" FILE *fopen(const char *file, const char *mode) {
  static int first = 0;
  if (__sync_lock_test_and_set(&first, 1) == 0)
    printf("user fopen\n");
  return __interceptor_fopen(file, mode);
}

extern "C" int fileno(FILE *f) {
  static int first = 0;
  if (__sync_lock_test_and_set(&first, 1) == 0)
    printf("user fileno\n");
  return 1;
}

int main() {
  FILE *f = fopen("/dev/zero", "r");
  if (f) {
    char buf;
    fread(&buf, 1, 1, f);
    fclose(f);
  }
}

// CHECK: user fopen
// CHECK-NOT: ThreadSanitizer

