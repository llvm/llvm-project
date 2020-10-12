// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SIGNED
// RUN: %clang_cc1 -ffixed-point -triple x86_64-unknown-linux-gnu -S -emit-llvm %s -o - -fpadding-on-unsigned-fixed-point | FileCheck %s --check-prefixes=CHECK,UNSIGNED

// Between different fixed point types
short _Accum sa_const = 2.5hk;
// CHECK-DAG: @sa_const  = {{.*}}global i16 320, align 2
_Accum a_const = 2.5hk;
// CHECK-DAG: @a_const   = {{.*}}global i32 81920, align 4
short _Accum sa_const2 = 2.5k;
// CHECK-DAG: @sa_const2 = {{.*}}global i16 320, align 2

short _Accum sa_from_f_const = 0.5r;
// CHECK-DAG: sa_from_f_const = {{.*}}global i16 64, align 2
_Fract f_from_sa_const = 0.5hk;
// CHECK-DAG: f_from_sa_const = {{.*}}global i16 16384, align 2

unsigned short _Accum usa_const = 2.5uk;
unsigned _Accum ua_const = 2.5uhk;
// SIGNED-DAG: @usa_const  = {{.*}}global i16 640, align 2
// SIGNED-DAG: @ua_const   = {{.*}}global i32 163840, align 4
// UNSIGNED-DAG:    @usa_const  = {{.*}}global i16 320, align 2
// UNSIGNED-DAG:    @ua_const   = {{.*}}global i32 81920, align 4

// FixedPoint to integer
int i_const = -128.0hk;
// CHECK-DAG: @i_const  = {{.*}}global i32 -128, align 4
int i_const2 = 128.0hk;
// CHECK-DAG: @i_const2 = {{.*}}global i32 128, align 4
int i_const3 = -128.0k;
// CHECK-DAG: @i_const3 = {{.*}}global i32 -128, align 4
int i_const4 = 128.0k;
// CHECK-DAG: @i_const4 = {{.*}}global i32 128, align 4
short s_const = -128.0k;
// CHECK-DAG: @s_const  = {{.*}}global i16 -128, align 2
short s_const2 = 128.0k;
// CHECK-DAG: @s_const2 = {{.*}}global i16 128, align 2

// Integer to fixed point
short _Accum sa_const5 = 2;
// CHECK-DAG: @sa_const5 = {{.*}}global i16 256, align 2
short _Accum sa_const6 = -2;
// CHECK-DAG: @sa_const6 = {{.*}}global i16 -256, align 2
short _Accum sa_const7 = -256;
// CHECK-DAG: @sa_const7 = {{.*}}global i16 -32768, align 2

// Signedness
unsigned short _Accum usa_const2 = 2.5hk;
// SIGNED-DAG: @usa_const2  = {{.*}}global i16 640, align 2
// UNSIGNED-DAG:    @usa_const2  = {{.*}}global i16 320, align 2
short _Accum sa_const3 = 2.5hk;
// CHECK-DAG: @sa_const3 = {{.*}}global i16 320, align 2

int i_const5 = 128.0uhk;
unsigned int ui_const = 128.0hk;
// CHECK-DAG: @i_const5  = {{.*}}global i32 128, align 4
// CHECK-DAG: @ui_const  = {{.*}}global i32 128, align 4

short _Accum sa_const9 = 2u;
// CHECK-DAG: @sa_const9 = {{.*}}global i16 256, align 2
unsigned short _Accum usa_const3 = 2;
// SIGNED-DAG: @usa_const3 = {{.*}}global i16 512, align 2
// UNSIGNED-DAG:    @usa_const3 = {{.*}}global i16 256, align 2

// Overflow (this is undefined but allowed)
short _Accum sa_const4 = 256.0k;
unsigned int ui_const2 = -2.5hk;
short _Accum sa_const8 = 256;
unsigned short _Accum usa_const4 = -2;

// Saturation
_Sat short _Accum sat_sa_const = 2.5hk;
// CHECK-DAG: @sat_sa_const  = {{.*}}global i16 320, align 2
_Sat short _Accum sat_sa_const2 = 256.0k;
// CHECK-DAG: @sat_sa_const2 = {{.*}}global i16 32767, align 2
_Sat unsigned short _Accum sat_usa_const = -1.0hk;
// CHECK-DAG: @sat_usa_const = {{.*}}global i16 0, align 2
_Sat unsigned short _Accum sat_usa_const2 = 256.0k;
// SIGNED-DAG: @sat_usa_const2 = {{.*}}global i16 -1, align 2
// UNSIGNED-DAG:    @sat_usa_const2 = {{.*}}global i16 32767, align 2

_Sat short _Accum sat_sa_const3 = 256;
// CHECK-DAG: @sat_sa_const3 = {{.*}}global i16 32767, align 2
_Sat short _Accum sat_sa_const4 = -257;
// CHECK-DAG: @sat_sa_const4 = {{.*}}global i16 -32768, align 2
_Sat unsigned short _Accum sat_usa_const3 = -1;
// CHECK-DAG: @sat_usa_const3 = {{.*}}global i16 0, align 2
_Sat unsigned short _Accum sat_usa_const4 = 256;
// SIGNED-DAG: @sat_usa_const4 = {{.*}}global i16 -1, align 2
// UNSIGNED-DAG:    @sat_usa_const4 = {{.*}}global i16 32767, align 2
