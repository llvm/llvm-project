// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SIGNED
// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -fpadding-on-unsigned-fixed-point -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,UNSIGNED

// Division between different fixed point types
short _Accum sa_const = 1.0hk / 2.0hk;
// CHECK-DAG: @sa_const  = {{.*}}global i16 64, align 2
_Accum a_const = 1.0hk / 2.0k;
// CHECK-DAG: @a_const   = {{.*}}global i32 16384, align 4
long _Accum la_const = 1.0hk / 2.0lk;
// CHECK-DAG: @la_const  = {{.*}}global i64 1073741824, align 8
short _Accum sa_const2 = 0.5hr / 2.0hk;
// CHECK-DAG: @sa_const2  = {{.*}}global i16 32, align 2
short _Accum sa_const3 = 0.5r / 2.0hk;
// CHECK-DAG: @sa_const3  = {{.*}}global i16 32, align 2
short _Accum sa_const4 = 0.5lr / 2.0hk;
// CHECK-DAG: @sa_const4  = {{.*}}global i16 32, align 2
short _Accum sa_const5 = 2.0hk / 0.5lr;
// CHECK-DAG: @sa_const5  = {{.*}}global i16 512, align 2

// Unsigned division
unsigned short _Accum usa_const = 3.0uhk / 2.0uhk;
// SIGNED-DAG:   @usa_const = {{.*}}global i16 384, align 2
// UNSIGNED-DAG: @usa_const = {{.*}}global i16 192, align 2

// Unsigned / signed
short _Accum sa_const6 = 1.0uhk / 2.0hk;
// CHECK-DAG: @sa_const6 = {{.*}}global i16 64, align 2

// Division with negative number
short _Accum sa_const7 = 0.5hr / (-2.0hk);
// CHECK-DAG: @sa_const7 = {{.*}}global i16 -32, align 2

// Int division
unsigned short _Accum usa_const2 = 2 / 0.5uhk;
// SIGNED-DAG:   @usa_const2 = {{.*}}global i16 1024, align 2
// UNSIGNED-DAG: @usa_const2 = {{.*}}global i16 512, align 2
short _Accum sa_const8 = 2 / (-0.5hk);
// CHECK-DAG: @sa_const8 = {{.*}}global i16 -512, align 2
short _Accum sa_const9 = 256 / 2.0hk;
// CHECK-DAG: @sa_const9 = {{.*}}global i16 16384, align 2
long _Fract lf_const = 0.5lr / -1;
// CHECK-DAG: @lf_const  = {{.*}}global i32 -1073741824, align 4

// Saturated division
_Sat short _Accum sat_sa_const = (_Sat short _Accum)128.0hk / (-0.25hk);
// CHECK-DAG: @sat_sa_const = {{.*}}global i16 -32768, align 2
_Sat unsigned short _Accum sat_usa_const = (_Sat unsigned short _Accum)128.0uhk / (0.25uhk);
// SIGNED-DAG:   @sat_usa_const = {{.*}}global i16 -1, align 2
// UNSIGNED-DAG: @sat_usa_const = {{.*}}global i16 32767, align 2
_Sat short _Accum sat_sa_const2 = (_Sat short _Accum)-128.0hk / (-0.0125hr);
// CHECK-DAG: @sat_sa_const2 = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const2 = (_Sat unsigned short _Accum)128.0uhk / (-128);
// CHECK-DAG:   @sat_usa_const2 = {{.*}}global i16 0, align 2
_Sat unsigned short _Accum sat_usa_const3 = (_Sat unsigned short _Accum)0.5uhk / -1;
// CHECK-DAG:   @sat_usa_const3 = {{.*}}global i16 0, align 2
_Sat short _Accum sat_sa_const3 = (_Sat short _Accum)-128.0hk / 128;
// CHECK-DAG: @sat_sa_const3 = {{.*}}global i16 -128, align 2
_Sat short _Accum sat_sa_const4 = (_Sat short _Accum)-25.7hk / 0.1lk;
// CHECK-DAG: @sat_sa_const4 = {{.*}}global i16 -32768, align 2

// Some more cases
short _Accum sa_const10 = 255.9921875hk / 255.9921875hk;
// CHECK-DAG: @sa_const10 = {{.*}}global i16 128, align 2
short _Accum sat_sa_const5 = (_Sat short _Accum)(-255.0hk - 1.0hk) / 0.0078125hk;
// CHECK-DAG: @sat_sa_const5 = {{.*}}global i16 -32768, align 2
_Sat short _Accum sat_sa_const6 = (_Sat short _Accum)(-255.0hk - 1.0hk) / -0.0078125hk;
// CHECK-DAG: @sat_sa_const6 = {{.*}}global i16 32767, align 2
short _Accum sa_const12 = 255.9921875hk / -1.0hk;
// CHECK-DAG: @sa_const12 = {{.*}}global i16 -32767, align 2
_Sat short _Accum sat_sa_const7 = (_Sat short _Accum)(-255.0hk - 1.0hk) / -1.0hk;
// CHECK-DAG: @sat_sa_const7 = {{.*}}global i16 32767, align 2
short _Accum sa_const13 = 0.0234375hk / 2.0hk;
// CHECK-DAG: @sa_const13 = {{.*}}global i16 1, align 2
short _Accum sa_const14 = -0.0234375hk / 2.0hk;
// CHECK-DAG: @sa_const14 = {{.*}}global i16 -2, align 2
short _Accum sa_const15 = -0.0078125hk / 255.28125hk;
// CHECK-DAG: @sa_const15 = {{.*}}global i16 -1, align 2
