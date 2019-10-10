// RUN: %clang -x c -fsanitize=pointer-overflow -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-C --implicit-check-not="error:"
// RUN: %clang -x c -fsanitize=pointer-overflow -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-C --implicit-check-not="error:"
// RUN: %clang -x c -fsanitize=pointer-overflow -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-C --implicit-check-not="error:"
// RUN: %clang -x c -fsanitize=pointer-overflow -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-C --implicit-check-not="error:"

// RUN: %clangxx    -fsanitize=pointer-overflow -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-CPP --implicit-check-not="error:"
// RUN: %clangxx    -fsanitize=pointer-overflow -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-CPP --implicit-check-not="error:"
// RUN: %clangxx    -fsanitize=pointer-overflow -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-CPP --implicit-check-not="error:"
// RUN: %clangxx    -fsanitize=pointer-overflow -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-CPP --implicit-check-not="error:"

#include <stdlib.h>

int main(int argc, char *argv[]) {
  char *base, *result;

  base = (char *)0;
  result = base + 0;
  // CHECK-C: {{.*}}.cpp:[[@LINE-1]]:17: runtime error: applying zero offset to null pointer

  base = (char *)0;
  result = base + 1;
  // CHECK: {{.*}}.cpp:[[@LINE-1]]:17: runtime error: applying non-zero offset 1 to null pointer

  base = (char *)1;
  result = base - 1;
  // CHECK: {{.*}}.cpp:[[@LINE-1]]:17: runtime error: applying non-zero offset to non-null pointer 0x{{.*}} produced null pointer

  return 0;
}
