// RUN: %clang_cc1 %s -triple arm64-apple-ios8.1.0 -std=c++11 -emit-llvm -o - | FileCheck %s
// rdar://20000762

typedef __attribute__((__ext_vector_type__(8))) float vector_float8;

typedef vector_float8 float8;

void MandelbrotPolyCalcSIMD8()
{
    constexpr float8   v4 = 4.0;  // value to compare against abs(z)^2, to see if bounded
    float8 vABS;
    auto vLT  = vABS < v4;
}

// CHECK: store <8 x float> 
// CHECK: [[ZERO:%.*]] = load <8 x float>, <8 x float>* [[VARBS:%.*]]
// CHECK: [[CMP:%.*]] = fcmp olt <8 x float> [[ZERO]]
// CHECK: [[SEXT:%.*]] = sext <8 x i1> [[CMP]] to <8 x i32>
// CHECK: store <8 x i32> [[SEXT]], <8 x i32>* [[VLT:%.*]]
