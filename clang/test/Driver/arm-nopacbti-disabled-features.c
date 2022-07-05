
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+nopacbti %s -### 2>&1 | FileCheck %s
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-m85+nopacbti %s -### 2>&1 | FileCheck %s

// CHECK-NOT: "-target-feature" "+pacbti"
// CHECK: "-target-feature" "-pacbti"
// CHECK-NOT: "-target-feature" "+pacbti"
