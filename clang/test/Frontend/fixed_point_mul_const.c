// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SIGNED
// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -fpadding-on-unsigned-fixed-point -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,UNSIGNED

// Multiplication between different fixed point types
short _Accum sa_const = 2.0hk * 2.0hk;
// CHECK-DAG: @sa_const  = {{.*}}global i16 512, align 2
_Accum a_const = 3.0hk * 2.0k;
// CHECK-DAG: @a_const   = {{.*}}global i32 196608, align 4
long _Accum la_const = 4.0hk * 2.0lk;
// CHECK-DAG: @la_const  = {{.*}}global i64 17179869184, align 8
short _Accum sa_const2 = 0.5hr * 2.0hk;
// CHECK-DAG: @sa_const2  = {{.*}}global i16 128, align 2
short _Accum sa_const3 = 0.5r * 3.0hk;
// CHECK-DAG: @sa_const3  = {{.*}}global i16 192, align 2
short _Accum sa_const4 = 0.5lr * 4.0hk;
// CHECK-DAG: @sa_const4  = {{.*}}global i16 256, align 2

// Unsigned multiplication
unsigned short _Accum usa_const = 1.0uhk * 2.0uhk;
// SIGNED-DAG:   @usa_const = {{.*}}global i16 512, align 2
// UNSIGNED-DAG: @usa_const = {{.*}}global i16 256, align 2

// Unsigned * signed
short _Accum sa_const5 = 20.0uhk * 3.0hk;
// CHECK-DAG: @sa_const5 = {{.*}}global i16 7680, align 2

// Multiplication with negative number
short _Accum sa_const6 = 0.5hr * (-2.0hk);
// CHECK-DAG: @sa_const6 = {{.*}}global i16 -128, align 2

// Int multiplication
unsigned short _Accum usa_const2 = 5 * 10.5uhk;
// SIGNED-DAG:   @usa_const2 = {{.*}}global i16 13440, align 2
// UNSIGNED-DAG: @usa_const2 = {{.*}}global i16 6720, align 2
short _Accum sa_const7 = 3 * (-0.5hk);
// CHECK-DAG: @sa_const7 = {{.*}}global i16 -192, align 2
short _Accum sa_const8 = 100 * (-2.0hk);
// CHECK-DAG: @sa_const8 = {{.*}}global i16 -25600, align 2
long _Fract lf_const = -0.25lr * 3;
// CHECK-DAG: @lf_const  = {{.*}}global i32 -1610612736, align 4

// Saturated multiplication
_Sat short _Accum sat_sa_const = (_Sat short _Accum)128.0hk * 3.0hk;
// CHECK-DAG: @sat_sa_const = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const = (_Sat unsigned short _Accum)128.0uhk * 128.0uhk;
// SIGNED-DAG:   @sat_usa_const = {{.*}}global i16 -1, align 2
// UNSIGNED-DAG: @sat_usa_const = {{.*}}global i16 32767, align 2
_Sat short _Accum sat_sa_const2 = (_Sat short _Accum)128.0hk * -128;
// CHECK-DAG: @sat_sa_const2 = {{.*}}global i16 -32768, align 2
_Sat unsigned short _Accum sat_usa_const2 = (_Sat unsigned short _Accum)128.0uhk * 30;
// SIGNED-DAG:   @sat_usa_const2 = {{.*}}global i16 -1, align 2
// UNSIGNED-DAG: @sat_usa_const2 = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const3 = (_Sat unsigned short _Accum)0.5uhk * (-2);
// CHECK-DAG:   @sat_usa_const3 = {{.*}}global i16 0, align 2
