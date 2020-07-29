// RUN: %clang -target arm-arm-none-eabi -mfloat-abi=soft %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-MFLOAT-ABI-SOFT
// CHECK-MFLOAT-ABI-SOFT: "-target-feature" "-dotprod"
// CHECK-MFLOAT-ABI-SOFT: "-target-feature" "-fp16fml"
// CHECK-MFLOAT-ABI-SOFT: "-target-feature" "-bf16"
// CHECK-MFLOAT-ABI-SOFT: "-target-feature" "-mve"
// CHECK-MFLOAT-ABI-SOFT: "-target-feature" "-mve.fp"
// CHECK-MFLOAT-ABI-SOFT: "-target-feature" "-fpregs"

// RUN: %clang -target arm-arm-none-eabi -mfpu=none %s -### 2>&1 | FileCheck %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8-a+nofp %s -### 2>&1 | FileCheck %s
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-a35+nofp %s -### 2>&1 | FileCheck %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8-a+nofp+nomve %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NOMVE
// RUN: %clang -target arm-arm-none-eabi -mcpu=cortex-a35+nofp+nomve %s -### 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NOMVE
// CHECK: "-target-feature" "-dotprod"
// CHECK: "-target-feature" "-fp16fml"
// CHECK: "-target-feature" "-bf16"
// CHECK: "-target-feature" "-mve.fp"
// CHECK-NOMVE: "-target-feature" "-fpregs"
