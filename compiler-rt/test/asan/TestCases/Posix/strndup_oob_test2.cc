// RUN: %clang_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

// When built as C on Linux, strndup is transformed to __strndup.
// RUN: %clang_asan -O3 -xc %s -o %t && not %run %t 2>&1 | FileCheck %s

// Unwind problem on arm: "main" is missing from the allocation stack trace.
// UNSUPPORTED: win32,s390,arm && !fast-unwinder-works

#include <string.h>

char kChars[] = { 'f', 'o', 'o' };

int main(int argc, char **argv) {
  char *copy = strndup(kChars, 3);
  copy = strndup(kChars, 10);
  // CHECK: AddressSanitizer: global-buffer-overflow
  // CHECK: {{.*}}main {{.*}}.cc:[[@LINE-2]]
  return *copy;
}
