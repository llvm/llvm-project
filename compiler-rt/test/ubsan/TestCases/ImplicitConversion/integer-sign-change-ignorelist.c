// RUN: %clang -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=implicit-integer-sign-change                           -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR
// RUN: %clang -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=implicit-integer-sign-change                           -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR
// RUN: %clang -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=implicit-integer-sign-change                           -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR
// RUN: %clang -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=implicit-integer-sign-change                           -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR

// RUN: rm -f %tmp
// RUN: echo "[implicit-integer-sign-change]" >> %tmp
// RUN: echo "fun:implicitSignChange" >> %tmp
// RUN: %clang -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=implicit-integer-sign-change -fsanitize-ignorelist=%tmp -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=implicit-integer-sign-change -fsanitize-ignorelist=%tmp -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=implicit-integer-sign-change -fsanitize-ignorelist=%tmp -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=implicit-integer-sign-change -fsanitize-ignorelist=%tmp -O3 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdint.h>
#include <stdio.h>

int32_t implicitSignChange(uint32_t argc) {
  fprintf(stderr, "TEST\n");
  // CHECK-NOT: runtime error
  // CHECK-LABEL: TEST
  return argc; // BOOM
  // ERROR: {{.*}}integer-sign-change-ignorelist.c:[[@LINE-1]]:10: runtime error: implicit conversion from type '{{.*}}' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type '{{.*}}' (aka 'int') changed the value to -1 (32-bit, signed)
  // CHECK-NOT: runtime error
}

int main(int argc, char **argv) {
  return !implicitSignChange(~0U);
}
