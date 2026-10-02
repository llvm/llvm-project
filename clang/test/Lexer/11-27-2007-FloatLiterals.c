// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// CHECK: f0x1FEC1E4A
// CHECK: 2.{{[0]*}}e+32
// CHECK: f0x1FEC1E4A
// CHECK: 2.{{[0]*}}e+32
// CHECK: +inf

float  F  = 1e-19f;
double D  = 2e32;
float  F2 = 01e-19f;
double D2 = 02e32;
float  F3 = 0xFp100000000000000000000F;
