// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation                           -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation                           -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation                           -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation                           -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR

// RUN: rm -f %tmp
// RUN: echo "[integer]" >> %tmp
// RUN: echo "fun:implicitUnsignedTruncation" >> %tmp
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O3 %s -o %t && %run %t 2>&1 | FileCheck %s

// RUN: rm -f %tmp
// RUN: echo "[implicit-conversion]" >> %tmp
// RUN: echo "fun:implicitUnsignedTruncation" >> %tmp
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O3 %s -o %t && %run %t 2>&1 | FileCheck %s

// RUN: rm -f %tmp
// RUN: echo "[implicit-integer-truncation]" >> %tmp
// RUN: echo "fun:implicitUnsignedTruncation" >> %tmp
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O3 %s -o %t && %run %t 2>&1 | FileCheck %s

// RUN: rm -f %tmp
// RUN: echo "[implicit-unsigned-integer-truncation]" >> %tmp
// RUN: echo "fun:implicitUnsignedTruncation" >> %tmp
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O3 %s -o %t && %run %t 2>&1 | FileCheck %s

// RUN: rm -f %tmp
// RUN: echo "[implicit-signed-integer-truncation]" >> %tmp
// RUN: echo "fun:implicitUnsignedTruncation" >> %tmp
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR
// RUN: %clang -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -fsanitize-ignorelist=%tmp -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,ERROR

#include <stdint.h>
#include <stdio.h>

uint8_t implicitUnsignedTruncation(uint32_t argc) {
  fprintf(stderr, "TEST\n");
  // CHECK-NOT: runtime error
  // CHECK-LABEL: TEST
  return argc; // BOOM
  // ERROR: {{.*}}unsigned-integer-truncation-ignorelist.c:[[@LINE-1]]:10: runtime error: implicit conversion from type '{{.*}}' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type '{{.*}}' (aka 'unsigned char') changed the value to 255 (8-bit, unsigned)
  // CHECK-NOT: runtime error
}

int main(int argc, char **argv) {
  return !implicitUnsignedTruncation(~0U);
}
