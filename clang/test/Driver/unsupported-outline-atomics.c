// RUN: %clang -target x86_64 -moutline-atomics -S %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-OUTLINE-ATOMICS-X86
// CHECK-OUTLINE-ATOMICS-X86: warning: 'x86_64' does not support '-moutline-atomics'; flag ignored [-Woption-ignored]
// CHECK-OUTLINE-ATOMICS-X86-NOT: "-target-feature" "+outline-atomics"

// RUN: %clang -target x86_64 -mno-outline-atomics -S %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-NO-OUTLINE-ATOMICS-X86
// CHECK-NO-OUTLINE-ATOMICS-X86: warning: 'x86_64' does not support '-mno-outline-atomics'; flag ignored [-Woption-ignored]
// CHECK-NO-OUTLINE-ATOMICS-X86-NOT: "-target-feature" "-outline-atomics"

// RUN: %clang -target riscv64 -moutline-atomics -S %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-OUTLINE-ATOMICS-RISCV
// CHECK-OUTLINE-ATOMICS-RISCV: warning: 'riscv64' does not support '-moutline-atomics'; flag ignored [-Woption-ignored]
// CHECK-OUTLINE-ATOMICS-RISCV-NOT: "-target-feature" "+outline-atomics"

// RUN: %clang -target riscv64 -mno-outline-atomics -S %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-NO-OUTLINE-ATOMICS-RISCV
// CHECK-NO-OUTLINE-ATOMICS-RISCV: warning: 'riscv64' does not support '-mno-outline-atomics'; flag ignored [-Woption-ignored]
// CHECK-NO-OUTLINE-ATOMICS-RISCV-NOT: "-target-feature" "-outline-atomics"
