// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple thumbv7-apple-darwin \
// RUN:   -disable-O0-optnone \
// RUN:   -target-cpu cortex-a8 \
// RUN:   -ffreestanding \
// RUN:   -emit-llvm -w -o - %s | opt -S -passes=mem2reg | FileCheck %s

#include <arm_neon.h>

uint8x8_t test_shift_vshr(uint8x8_t a) {
  // CHECK-LABEL: test_shift_vshr
  // CHECK: %{{.*}} = lshr <8 x i8> %a, <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  return vshr_n_u8(a, 5);
}

int8x8_t test_shift_vshr_smax(int8x8_t a) {
  // CHECK-LABEL: test_shift_vshr_smax
  // CHECK: %{{.*}} = ashr <8 x i8> %a, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  return vshr_n_s8(a, 8);
}

uint8x8_t test_shift_vshr_umax(uint8x8_t a) {
  // CHECK-LABEL: test_shift_vshr_umax
  // CHECK: ret <8 x i8> zeroinitializer
  return vshr_n_u8(a, 8);
}

uint8x8_t test_shift_vsra(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_shift_vsra
  // CHECK: %[[SHR:.*]] = lshr <8 x i8> %b, <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>
  // CHECK: %{{.*}} = add <8 x i8> %a, %[[SHR]]
  return vsra_n_u8(a, b, 5);
}

int8x8_t test_shift_vsra_smax(int8x8_t a, int8x8_t b) {
  // CHECK-LABEL: test_shift_vsra_smax
  // CHECK: %[[SHR:.*]] = ashr <8 x i8> %b, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  // CHECK: %{{.*}} = add <8 x i8> %a, %[[SHR]]
  return vsra_n_s8(a, b, 8);
}

uint8x8_t test_shift_vsra_umax(uint8x8_t a, uint8x8_t b) {
  // CHECK-LABEL: test_shift_vsra_umax
  // CHECK: [[RES:%.*]] = add <8 x i8> %a, zeroinitializer
  // CHECK: ret <8 x i8> [[RES]]
  return vsra_n_u8(a, b, 8);
}
