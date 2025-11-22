// RUN: %clang_cc1 -triple arm-linux-guneabi \
// RUN:   -target-cpu cortex-a8 \
// RUN:   -emit-llvm -w -O1 -o - %s | FileCheck --check-prefix=CHECK-ARM %s

// RUN: %clang_cc1 -triple arm64-linux-gnueabi \
// RUN:   -target-feature +neon \
// RUN:   -emit-llvm -w -O1 -o - %s | FileCheck --check-prefix=CHECK-AARCH64 %s

// RUN: %clang_cc1 -triple arm-linux-guneabi \
// RUN:   -target-cpu cortex-a8 -fexperimental-new-constant-interpreter \
// RUN:   -emit-llvm -w -O1 -o - %s | FileCheck --check-prefix=CHECK-ARM %s

// RUN: %clang_cc1 -triple arm64-linux-gnueabi \
// RUN:   -target-feature +neon -fexperimental-new-constant-interpreter \
// RUN:   -emit-llvm -w -O1 -o - %s | FileCheck --check-prefix=CHECK-AARCH64 %s

// REQUIRES: aarch64-registered-target || arm-registered-target

// Test if int64_t and uint64_t can be correctly mangled.

#include "arm_neon.h"
// CHECK-ARM: f1x(
// CHECK-AARCH64: f1l(
void f1(int64_t a) {}
// CHECK-ARM: f2y(
// CHECK-AARCH64: f2m(
void f2(uint64_t a) {}
// CHECK-ARM: f3Px(
// CHECK-AARCH64: f3Pl(
void f3(int64_t *ptr) {}
// CHECK-ARM: f4Py(
// CHECK-AARCH64: f4Pm(
void f4(uint64_t *ptr) {}
