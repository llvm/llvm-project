// RUN: %clang_cc1 -ffixed-point -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SIGNED
// RUN: %clang_cc1 -ffixed-point -fpadding-on-unsigned-fixed-point -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,UNSIGNED

short _Accum sa_const1 = 1.0hk << 2;       // CHECK-DAG: @sa_const1 = {{.*}}global i16 512
short _Accum sa_const2 = 0.5hk << 2;       // CHECK-DAG: @sa_const2 = {{.*}}global i16 256
short _Accum sa_const3 = 10.0hk >> 3;      // CHECK-DAG: @sa_const3 = {{.*}}global i16 160
short _Accum sa_const4 = 0.0546875hk << 8; // CHECK-DAG: @sa_const4 = {{.*}}global i16 1792
short _Accum sa_const5 = -1.0hk << 2;      // CHECK-DAG: @sa_const5 = {{.*}}global i16 -512
short _Accum sa_const6 = -255.0hk >> 8;    // CHECK-DAG: @sa_const6 = {{.*}}global i16 -128

_Fract f_const1 = -1.0r >> 5;              // CHECK-DAG: @f_const1 = {{.*}}global i16 -1024
_Fract f_const2 = 0.0052490234375r >> 3;   // CHECK-DAG: @f_const2 = {{.*}}global i16 21
_Fract f_const3 = -0.0001r << 5;           // CHECK-DAG: @f_const3 = {{.*}}global i16 -96
_Fract f_const4 = -0.75r >> 15;            // CHECK-DAG: @f_const4 = {{.*}}global i16 -1
_Fract f_const5 = 0.078216552734375r << 3; // CHECK-DAG: @f_const5 = {{.*}}global i16 20504

unsigned _Fract uf_const1 = 0.375ur >> 13;
// SIGNED-DAG:   @uf_const1 = {{.*}}global i16 3
// UNSIGNED-DAG: @uf_const1 = {{.*}}global i16 1
unsigned _Fract uf_const2 = 0.0546875ur << 3;
// SIGNED-DAG:   @uf_const2 = {{.*}}global i16 28672
// UNSIGNED-DAG: @uf_const2 = {{.*}}global i16 14336

_Sat short _Accum ssa_const1 = (_Sat short _Accum)31.875hk << 4; // CHECK-DAG: @ssa_const1 = {{.*}}global i16 32767
_Sat short _Accum ssa_const2 = (_Sat short _Accum) - 1.0hk << 8; // CHECK-DAG: @ssa_const2 = {{.*}}global i16 -32768
_Sat short _Accum ssa_const3 = (_Sat short _Accum)128.0hk << 8;  // CHECK-DAG: @ssa_const3 = {{.*}}global i16 32767
_Sat short _Fract ssf_const1 = (_Sat short _Fract) - 0.5hr << 3; // CHECK-DAG: @ssf_const1 = {{.*}}global i8 -128

_Sat unsigned _Fract suf_const1 = (_Sat unsigned _Fract)0.5r << 1;
// SIGNED-DAG:   @suf_const1 = {{.*}}global i16 -1
// UNSIGNED-DAG: @suf_const1 = {{.*}}global i16 32767
_Sat unsigned _Fract suf_const2 = (_Sat unsigned _Fract)0.25r << 1;
// SIGNED-DAG:   @suf_const2 = {{.*}}global i16 -32768
// UNSIGNED-DAG: @suf_const2 = {{.*}}global i16 16384
_Sat unsigned _Accum sua_const2 = (_Sat unsigned _Accum)128.0uk << 10;
// SIGNED-DAG:   @sua_const2 = {{.*}}global i32 -1
// UNSIGNED-DAG: @sua_const2 = {{.*}}global i32 2147483647
