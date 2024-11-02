// RUN: %clang_cc1 -triple aarch64-arm-unknown-eabi -Wno-int-conversion -target-feature +mops -target-feature +mte -w -S -emit-llvm -o - %s  | FileCheck --check-prefix=CHECK-MOPS   %s
// RUN: not %clang_cc1 -triple aarch64-arm-unknown-eabi -Wno-int-conversion -target-feature +mops -Wno-implicit-function-declaration -w -S -emit-llvm -o - %s 2>&1  | FileCheck --check-prefix=CHECK-NOMOPS %s
// RUN: not %clang_cc1 -triple aarch64-arm-unknown-eabi -Wno-int-conversion -Wno-implicit-function-declaration -target-feature +mte -w -S -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=CHECK-NOMOPS %s
// RUN: not %clang_cc1 -triple aarch64-arm-unknown-eabi -Wno-int-conversion -Wno-implicit-function-declaration -w -S -emit-llvm -o - %s 2>&1 | FileCheck --check-prefix=CHECK-NOMOPS %s

#include <arm_acle.h>
#include <stddef.h>

// CHECK-LABEL:       @bzero_0(
// CHECK-MOPS:        @llvm.aarch64.mops.memset.tag
// CHECK-NOMOPS:      '__builtin_arm_mops_memset_tag' needs target feature mte,mops
void *bzero_0(void *dst) {
  return __arm_mops_memset_tag(dst, 0, 0);
}

// CHECK-LABEL:       @bzero_1(
// CHECK-MOPS:        @llvm.aarch64.mops.memset.tag
void *bzero_1(void *dst) {
  return __arm_mops_memset_tag(dst, 0, 1);
}

// CHECK-LABEL:       @bzero_10(
// CHECK-MOPS:        @llvm.aarch64.mops.memset.tag
void *bzero_10(void *dst) {
  return __arm_mops_memset_tag(dst, 0, 10);
}

// CHECK-LABEL:       @bzero_10000(
// CHECK-MOPS:        @llvm.aarch64.mops.memset.tag
void *bzero_10000(void *dst) {
  return __arm_mops_memset_tag(dst, 0, 10000);
}

// CHECK-LABEL:       @bzero_n(
// CHECK-MOPS:        @llvm.aarch64.mops.memset.tag
void *bzero_n(void *dst, size_t size) {
  return __arm_mops_memset_tag(dst, 0, size);
}

// CHECK-LABEL:       @memset_0(
// CHECK-MOPS:        @llvm.aarch64.mops.memset.tag
void *memset_0(void *dst, int value) {
  return __arm_mops_memset_tag(dst, value, 0);
}

// CHECK-LABEL:       @memset_1(
// CHECK-MOPS:        @llvm.aarch64.mops.memset.tag
void *memset_1(void *dst, int value) {
  return __arm_mops_memset_tag(dst, value, 1);
}

// CHECK-LABEL:       @memset_10(
// CHECK-MOPS:        @llvm.aarch64.mops.memset.tag
void *memset_10(void *dst, int value) {
  return __arm_mops_memset_tag(dst, value, 10);
}

// CHECK-LABEL:       @memset_10000(
// CHECK-MOPS:        @llvm.aarch64.mops.memset.tag
void *memset_10000(void *dst, int value) {
  return __arm_mops_memset_tag(dst, value, 10000);
}

// CHECK-LABEL:       @memset_n(
// CHECK-MOPS:        @llvm.aarch64.mops.memset.tag
void *memset_n(void *dst, int value, size_t size) {
  return __arm_mops_memset_tag(dst, value, size);
}
