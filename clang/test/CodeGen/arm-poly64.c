// RUN: %clang_cc1 -triple armv8.2a-arm-none-eabi -target-feature +neon \
// RUN:  -emit-llvm -o - %s | FileCheck %s

// Test that we can use the poly64 type on AArch32

#include <arm_neon.h>

// CHECK-LABEL: @test_poly64
// CHECK: ret i64 %0
poly64_t test_poly64(poly64_t a) {
  return a;
}
