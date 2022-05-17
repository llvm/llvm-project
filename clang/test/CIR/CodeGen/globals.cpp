// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

int a = 3;
const int b = 4; // unless used wont be generated

unsigned long int c = 2;
float y = 3.4;
double w = 4.3;
char x = '3';
unsigned char rgb[3] = {0, 233, 33};
char alpha[4] = "abc";
const char *s = "example";

// CHECK: module  {
// CHECK-NEXT: cir.global @a = 3 : i32
// CHECK-NEXT: cir.global @c = 2 : i64
// CHECK-NEXT: cir.global @y = 3.400000e+00 : f32
// CHECK-NEXT: cir.global @w = 4.300000e+00 : f64
// CHECK-NEXT: cir.global @x = 51 : i8
// CHECK-NEXT: cir.global @rgb = #cir.cst_array<[0 : i8, -23 : i8, 33 : i8] : !cir.array<i8 x 3>>
// CHECK-NEXT: cir.global @alpha = #cir.cst_array<[97 : i8, 98 : i8, 99 : i8, 0 : i8] : !cir.array<i8 x 4>>

// CHECK-NEXT: cir.global "private" constant @".str" = #cir.cst_array<"example\00" : !cir.array<i8 x 8>> : !cir.array<i8 x 8> {alignment = 1 : i64}
// CHECK-NEXT: cir.global @s = @".str": !cir.ptr<i8>
