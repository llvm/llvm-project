// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-linux-gnu | FileCheck %s

extern int aa __attribute__((section(".sdata")));
// CHECK-DAG: @aa = external global i32, section ".sdata", align 4

extern int bb __attribute__((section(".sdata"))) = 1;
// CHECK-DAG: @bb ={{.*}} global i32 1, section ".sdata", align 4

int foo() {
  return aa + bb;
}
