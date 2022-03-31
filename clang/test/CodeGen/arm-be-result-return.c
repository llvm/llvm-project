// RUN: %clang_cc1 -triple armebv7-arm-none-eabi -emit-llvm -w -o - %s | FileCheck %s

// this tests for AAPCS section 5.4:
// A Composite Type not larger than 4 bytes is returned in r0.
// The format is as if the result had been stored in memory at a
// word-aligned address and then loaded into r0 with an LDR instruction

extern union Us { short s; } us;
union Us callee_us(void) { return us; }
// CHECK-LABEL: callee_us()
// CHECK: zext i16
// CHECK: shl 
// CHECK: ret i32

void caller_us(void) {
  us = callee_us();
// CHECK-LABEL: caller_us()
// CHECK: call i32
// CHECK: lshr i32
// CHECK: trunc i32
}

extern struct Ss { short s; } ss;
struct Ss callee_ss(void) { return ss; }
// CHECK-LABEL: callee_ss()
// CHECK: zext i16
// CHECK: shl 
// CHECK: ret i32

void caller_ss(void) {
  ss = callee_ss();
// CHECK-LABEL: caller_ss()
// CHECK: call i32
// CHECK: lshr i32
// CHECK: trunc i32
}

