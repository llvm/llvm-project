// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int a = 3;
const int b = 4; // unless used wont be generated

unsigned long int c = 2;
float y = 3.4;
double w = 4.3;

// CHECK: module  {
// CHECK-NEXT: cir.global @a : i32 = 3
// CHECK-NEXT: cir.global @c : i64 = 2
// CHECK-NEXT: cir.global @y : f32 = 3.400000e+00
// CHECK-NEXT: cir.global @w : f64 = 4.300000e+00