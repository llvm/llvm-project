// RUN: not %clang -target arm-arm-eabi -march=armv8-a+bf16 -mfloat-abi=soft -c %s 2>&1 | FileCheck %s

// CHECK: error: __bf16 is not supported on this target
extern __bf16 var;
