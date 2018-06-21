// RUN: %clang -target aarch64-linux-eabi %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DEF %s
// RUN: %clang -target aarch64-linux-eabi -mfix-cortex-a53-835769 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-YES %s
// RUN: %clang -target aarch64-linux-eabi -mno-fix-cortex-a53-835769 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO %s

// RUN: %clang -target aarch64-android-eabi %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-YES %s

// CHECK-DEF-NOT: "-backend-option" "-aarch64-fix-cortex-a53-835769"
// CHECK-YES: "-backend-option" "-aarch64-fix-cortex-a53-835769=1"
// CHECK-NO: "-backend-option" "-aarch64-fix-cortex-a53-835769=0"
