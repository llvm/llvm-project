// Test the -mgeneral-regs-only option

// RUN: %clang --target=aarch64-linux-eabi -mgeneral-regs-only %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FP %s
// RUN: %clang --target=arm64-linux-eabi -mgeneral-regs-only %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FP %s
// CHECK-NO-FP: "-target-feature" "-fp-armv8"
// CHECK-NO-FP: "-target-feature" "-neon"
