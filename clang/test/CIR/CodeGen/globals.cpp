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
const char *s1 = "example1";
const char *s2 = "example";

void use_global() {
  int li = a;
}

void use_global_string() {
  unsigned char c = s2[0];
}

// CHECK: module {{.*}} {
// CHECK-NEXT: cir.global external @a = 3 : i32
// CHECK-NEXT: cir.global external @c = 2 : i64
// CHECK-NEXT: cir.global external @y = 3.400000e+00 : f32
// CHECK-NEXT: cir.global external @w = 4.300000e+00 : f64
// CHECK-NEXT: cir.global external @x = 51 : i8
// CHECK-NEXT: cir.global external @rgb = #cir.cst_array<[0 : i8, -23 : i8, 33 : i8] : !cir.array<i8 x 3>>
// CHECK-NEXT: cir.global external @alpha = #cir.cst_array<[97 : i8, 98 : i8, 99 : i8, 0 : i8] : !cir.array<i8 x 4>>

// CHECK-NEXT: cir.global "private" constant internal @".str" = #cir.cst_array<"example\00" : !cir.array<i8 x 8>> : !cir.array<i8 x 8> {alignment = 1 : i64}
// CHECK-NEXT: cir.global external @s = @".str": !cir.ptr<i8>

// CHECK-NEXT: cir.global "private" constant internal @".str1" = #cir.cst_array<"example1\00" : !cir.array<i8 x 9>> : !cir.array<i8 x 9> {alignment = 1 : i64}
// CHECK-NEXT: cir.global external @s1 = @".str1": !cir.ptr<i8>

// CHECK-NEXT: cir.global external @s2 = @".str": !cir.ptr<i8>

// CHECK: cir.func @_Z10use_globalv() {
// CHECK-NEXT:     %0 = cir.alloca i32, cir.ptr <i32>, ["li", init] {alignment = 4 : i64}
// CHECK-NEXT:     %1 = cir.get_global @a : cir.ptr <i32>
// CHECK-NEXT:     %2 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:     cir.store %2, %0 : i32, cir.ptr <i32>

// CHECK: cir.func @_Z17use_global_stringv() {
// CHECK-NEXT:   %0 = cir.alloca i8, cir.ptr <i8>, ["c", init] {alignment = 1 : i64}
// CHECK-NEXT:   %1 = cir.get_global @s2 : cir.ptr <!cir.ptr<i8>>
// CHECK-NEXT:   %2 = cir.load %1 : cir.ptr <!cir.ptr<i8>>, !cir.ptr<i8>
// CHECK-NEXT:   %3 = cir.cst(0 : i32) : i32
// CHECK-NEXT:   %4 = cir.ptr_stride(%2 : !cir.ptr<i8>, %3 : i32), !cir.ptr<i8>
// CHECK-NEXT:   %5 = cir.load %4 : cir.ptr <i8>, i8
// CHECK-NEXT:   cir.store %5, %0 : i8, cir.ptr <i8>
