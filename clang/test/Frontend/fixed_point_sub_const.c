// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SIGNED
// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -fpadding-on-unsigned-fixed-point -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,UNSIGNED

// Subtraction between different fixed point types
short _Accum sa_const = 1.0hk - 2.0hk;
// CHECK-DAG: @sa_const  = {{.*}}global i16 -128, align 2
_Accum a_const = 1.0hk - 2.0k;
// CHECK-DAG: @a_const   = {{.*}}global i32 -32768, align 4
long _Accum la_const = 1.0hk - 2.0lk;
// CHECK-DAG: @la_const  = {{.*}}global i64 -2147483648, align 8
short _Accum sa_const2 = 0.5hr - 2.0hk;
// CHECK-DAG: @sa_const2  = {{.*}}global i16 -192, align 2
short _Accum sa_const3 = 0.5r - 2.0hk;
// CHECK-DAG: @sa_const3  = {{.*}}global i16 -192, align 2
short _Accum sa_const4 = 0.5lr - 2.0hk;
// CHECK-DAG: @sa_const4  = {{.*}}global i16 -192, align 2
short _Accum sa_const5 = 2.0hk - 0.5lr;
// CHECK-DAG: @sa_const5  = {{.*}}global i16 192, align 2

// Unsigned subtraction
unsigned short _Accum usa_const = 3.0uhk - 2.0uhk;
// SIGNED-DAG:   @usa_const = {{.*}}global i16 256, align 2
// UNSIGNED-DAG: @usa_const = {{.*}}global i16 128, align 2

// Unsigned - signed
short _Accum sa_const6 = 1.0uhk - 2.0hk;
// CHECK-DAG: @sa_const6 = {{.*}}global i16 -128, align 2

// Subtraction with negative number
short _Accum sa_const7 = 0.5hr - (-2.0hk);
// CHECK-DAG: @sa_const7 = {{.*}}global i16 320, align 2

// Int subtraction
unsigned short _Accum usa_const2 = 2 - 0.5uhk;
// SIGNED-DAG:   @usa_const2 = {{.*}}global i16 384, align 2
// UNSIGNED-DAG: @usa_const2 = {{.*}}global i16 192, align 2
short _Accum sa_const8 = 2 - (-0.5hk);
// CHECK-DAG: @sa_const8 = {{.*}}global i16 320, align 2
short _Accum sa_const9 = 257 - 2.0hk;
// CHECK-DAG: @sa_const9 = {{.*}}global i16 32640, align 2
long _Fract lf_const = 0.5lr - 1;
// CHECK-DAG: @lf_const  = {{.*}}global i32 -1073741824, align 4

// Saturated subtraction
_Sat short _Accum sat_sa_const = (_Sat short _Accum)128.0hk - (-128.0hk);
// CHECK-DAG: @sat_sa_const = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const = (_Sat unsigned short _Accum)128.0uhk - (-128.0uhk);
// CHECK-DAG: @sat_usa_const = {{.*}}global i16 0, align 2
_Sat short _Accum sat_sa_const2 = (_Sat short _Accum)128.0hk - (-128);
// CHECK-DAG: @sat_sa_const2 = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const2 = (_Sat unsigned short _Accum)128.0uhk - (-128);
// SIGNED-DAG:   @sat_usa_const2 = {{.*}}global i16 -1, align 2
// UNSIGNED-DAG: @sat_usa_const2 = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const3 = (_Sat unsigned short _Accum)0.5uhk - 2;
// CHECK-DAG:   @sat_usa_const3 = {{.*}}global i16 0, align 2
_Sat short _Accum sat_sa_const3 = (_Sat short _Accum)-128.0hk - 128;
// CHECK-DAG: @sat_sa_const3 = {{.*}}global i16 -32768, align 2
_Sat short _Accum sat_sa_const4 = (_Sat short _Accum)-150.0hk - 130.0lk;
// CHECK-DAG: @sat_sa_const4 = {{.*}}global i16 -32768, align 2
