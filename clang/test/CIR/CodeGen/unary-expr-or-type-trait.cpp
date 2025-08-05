// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

void foo() {
  unsigned long b = sizeof(bool);
  // CHECK: cir.const #cir.int<1> : !u64i

  unsigned long i = sizeof(int);
  // CHECK: cir.const #cir.int<4> : !u64i

  unsigned long l =  sizeof(long);
  // CHECK: cir.const #cir.int<8> : !u64i

  unsigned long f =  sizeof(float);
  // CHECK: cir.const #cir.int<4> : !u64i

  unsigned long d =  sizeof(double);
  // CHECK: cir.const #cir.int<8> : !u64i

  unsigned long iArr = sizeof(int[5]);
  // CHECK: cir.const #cir.int<20> : !u64i

  unsigned long dArr =  sizeof(double[5]);
  // CHECK: cir.const #cir.int<40> : !u64i
}

void foo2() {
  unsigned long b = alignof(bool);
  // CHECK: cir.const #cir.int<1> : !u64i

  unsigned long i = alignof(int);
  // CHECK: cir.const #cir.int<4> : !u64i

  unsigned long l =  alignof(long);
  // CHECK: cir.const #cir.int<8> : !u64i

  unsigned long f =  alignof(float);
  // CHECK: cir.const #cir.int<4> : !u64i

  unsigned long d =  alignof(double);
  // CHECK: cir.const #cir.int<8> : !u64i

  unsigned long iArr = alignof(int[5]);
  // CHECK: cir.const #cir.int<4> : !u64i

  unsigned long dArr =  alignof(double[5]);
  // CHECK: cir.const #cir.int<8> : !u64i
}
