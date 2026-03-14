// REQUIRES: arm-registered-target

// Check that Cortex-M cores don't enable hwdiv-arm (and don't emit Tag_DIV_use)
//
// This target feature doesn't affect C predefines, nor the generated IR;
// only the build attributes in generated assembly and object files are affected.

// RUN: %clang -target arm-linux-gnueabi -mcpu=cortex-m3 -S %s -o - | FileCheck %s
// RUN: %clang -target arm-linux-gnueabi -mcpu=cortex-m4 -S %s -o - | FileCheck %s
// CHECK-NOT: .eabi_attribute	44

