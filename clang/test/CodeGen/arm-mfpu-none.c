// REQUIRES: arm-registered-target
// RUN: %clang -target arm-none-eabi -mcpu=cortex-m4 -mfpu=none -S -o - %s | FileCheck %s

// CHECK-LABEL: compute
// CHECK-NOT: {{s[0-9]}}
// CHECK: .fnend
float compute(float a, float b) {
  return (a+b) * (a-b);
}
