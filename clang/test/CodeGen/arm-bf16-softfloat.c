// REQUIRES: arm-registered-target
// RUN: not %clang -target arm-arm-none-eabi -march=armv8-a+bf16 -mfloat-abi=soft -c %s -o %t 2>&1 | FileCheck %s
// RUN: not %clang -target arm-arm-none-eabi -march=armv8-a+bf16 -mfpu=none -c %s -o %t 2>&1 | FileCheck %s
// RUN: not %clang -target arm-arm-none-eabi -march=armv8-a+bf16+nofp -c %s -o %t 2>&1 | FileCheck %s
// RUN: not %clang -target arm-arm-none-eabi -march=armv8-a+bf16+fp+nofp -c %s -o %t 2>&1 | FileCheck %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8-a+bf16+fp -c %s -o %t
// RUN: %clang -target arm-arm-none-eabi -march=armv8-a+bf16+nofp+fp -c %s -o %t

// CHECK: error: __bf16 is not supported on this target
extern __bf16 var;
