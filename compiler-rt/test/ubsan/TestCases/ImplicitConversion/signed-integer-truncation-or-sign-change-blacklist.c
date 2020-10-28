// UNSUPPORTED: ios

// All of these don't actually silence it:

// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change                           -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change                           -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change                           -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change                           -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"

// RUN: rm -f %tmp
// RUN: echo "[implicit-signed-integer-truncation]" >> %tmp
// RUN: echo "fun:implicitConversion" >> %tmp
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"

// RUN: rm -f %tmp
// RUN: echo "[implicit-signed-integer-truncation,implicit-integer-sign-change]" >> %tmp
// RUN: echo "fun:implicitConversion" >> %tmp
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion"


// The only two way it works:

// RUN: rm -f %tmp
// RUN: echo "[implicit-integer-sign-change]" >> %tmp
// RUN: echo "fun:implicitConversion" >> %tmp
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O0 %s -o %t && %run %t 2>&1 | count 0
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O1 %s -o %t && %run %t 2>&1 | count 0
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O2 %s -o %t && %run %t 2>&1 | count 0
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O3 %s -o %t && %run %t 2>&1 | count 0

// RUN: rm -f %tmp
// RUN: echo "[implicit-signed-integer-truncation]" >> %tmp
// RUN: echo "[implicit-integer-sign-change]" >> %tmp
// RUN: echo "fun:implicitConversion" >> %tmp
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O0 %s -o %t && %run %t 2>&1 | count 0
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O1 %s -o %t && %run %t 2>&1 | count 0
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O2 %s -o %t && %run %t 2>&1 | count 0
// RUN: %clang -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-blacklist=%tmp -O3 %s -o %t && %run %t 2>&1 | count 0

#include <stdint.h>

int8_t implicitConversion(uint32_t argc) {
  return argc; // BOOM
// CHECK: {{.*}}signed-integer-truncation-or-sign-change-blacklist.c:[[@LINE-1]]:10: runtime error: implicit conversion from type '{{.*}}' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type '{{.*}}' (aka '{{(signed )?}}char') changed the value to -1 (8-bit, signed)
}

int main(int argc, char **argv) {
  return !implicitConversion(~0U);
}
