// RUN: %clang --target=arm-none-eabi -x c - -o - -S < %s -mcpu=cortex-a5 -mfpu=vfpv4-d16 | FileCheck %s
// REQUIRES: arm-registered-target
// CHECK: .fpu vfpv4-d16
void foo() {}
