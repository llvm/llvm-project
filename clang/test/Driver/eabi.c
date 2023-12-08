// RUN: %clang %s -meabi 4 -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-EABI4 %s
// RUN: %clang %s -meabi 5 -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-EABI5 %s
// RUN: %clang %s -meabi gnu -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-GNUEABI %s
// RUN: not %clang %s -meabi unknown 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-UNKNOWN %s

// CHECK-EABI4: "-meabi" "4"
// CHECK-EABI5: "-meabi" "5"
// CHECK-GNUEABI: "-meabi" "gnu"
// CHECK-UNKNOWN: error: invalid value 'unknown' in '-meabi unknown'
