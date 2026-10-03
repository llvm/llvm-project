// -meabi=4 has no triple environment, so it is forwarded to cc1.
// RUN: %clang %s -target arm-none-eabi -meabi 4 -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-EABI4 %s

// -meabi=gnu/5 are encoded in the cc1 -triple environment, not forwarded.
// RUN: %clang %s -target arm-none-eabi -meabi gnu -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-GNUEABI %s
// RUN: %clang %s -target arm-none-gnueabi -meabi 5 -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-EABI5 %s

// RUN: not %clang %s -meabi unknown 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-UNKNOWN %s

// CHECK-EABI4: "-triple" "armv{{.*}}-unknown-none-eabi"
// CHECK-EABI4: "-meabi" "4"
// CHECK-GNUEABI: "-triple" "armv{{.*}}-unknown-none-gnueabi"
// CHECK-GNUEABI-NOT: "-meabi"
// CHECK-EABI5: "-triple" "armv{{.*}}-unknown-none-eabi"
// CHECK-EABI5-NOT: "-meabi"
// CHECK-UNKNOWN: error: invalid value 'unknown' in '-meabi unknown'
