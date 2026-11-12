// RUN: %clang_cc1 -fgnuc-version=4.2.1 -std=c11 -E -dM %s 2>&1 | FileCheck --check-prefix=CHECK-C11 %s
// RUN: %clang_cc1 -fgnuc-version=4.2.1 -std=c2x -E -dM %s 2>&1 | FileCheck --check-prefix=CHECK-C2X %s

#include <stdbool.h>

// CHECK-C11: #define bool _Bool
// CHECK-C11: #define false 0
// CHECK-C11: #define true 1

// CHECK-C2X-NOT: #define bool
// CHECK-C2X-NOT: #define true
// CHECK-C2X-NOT: #define false
