// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -disable-O0-optnone -emit-llvm -o - %s 2>&1 | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// Check float conversion is not accepted for unsigned int argument
int8_t test_vuqaddb_s8(){
  return vuqaddb_s8(1, -1.0f);
}

int16_t test_vuqaddh_s16() {
  return vuqaddh_s16(1, -1.0f);
}

int32_t test_vuqadds_s32() {
  return vuqadds_s32(1, -1.0f);
}

int64_t test_vuqaddd_s64() {
  return vuqaddd_s64(1, -1.0f);
}
// CHECK: warning: implicit conversion of out of range value from 'float' to 'uint8_t' (aka 'unsigned char') is undefined
// CHECK: warning: implicit conversion of out of range value from 'float' to 'uint16_t' (aka 'unsigned short') is undefined
// CHECK: warning: implicit conversion of out of range value from 'float' to 'uint32_t' (aka 'unsigned int') is undefined
// CHECK: warning: implicit conversion of out of range value from 'float' to 'uint64_t' (aka 'unsigned long') is undefined

