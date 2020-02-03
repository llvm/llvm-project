// RUN: %clang_cc1 -std=c++11 -S -emit-llvm -o - %s -triple x86_64-linux-gnu | FileCheck %s

typedef __attribute__((vector_size(4*4))) float float32x4_t;
union QDSUnion { float32x4_t q; float s[4]; };
constexpr float32x4_t a = {1,2,3,4};
QDSUnion t = {{(a)}};
// CHECK: @t = global %union.QDSUnion { <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00> }
