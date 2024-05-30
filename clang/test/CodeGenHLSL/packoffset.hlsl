// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: @A.cb. = external constant { [8 x i8], double, [8 x i8], float, float, half, i16, [4 x i8], i64, i32 }
cbuffer A {
  float a : packoffset(c1.z);
  double b : packoffset(c0.z);
  float  c;
  half   d;
  int16_t e;
  int64_t f;
  int     g;
}

// CHECK: @B.cb. = external constant { [24 x i8], float, [4 x i8], double, [8 x i8], <3 x float>, [4 x i8], <3 x double>, half, [6 x i8], <2 x double>, float, <3 x half>, <3 x half> }
cbuffer B {
  double  B0;
  float3  B1;
  float   B2 : packoffset(c1.z);
  double3 B3;
  half    B4;
  double2 B5;
  float   B6;
  half3   B7;
  half3   B8;
}

// CHECK: @C.cb. = external constant { [2 x double], [8 x i8], [3 x <3 x float>], float, [3 x double], half, [6 x i8], [1 x <2 x double>], float, [12 x i8], [2 x <3 x half>], <3 x half> }
cbuffer C {
  double C0[2] : packoffset(c0);
  float3 C1[3];
  float  C2;
  double C3[3];
  half   C4;
  double2 C5[1];
  float  C6;
  half3  C7[2];
  half3  C8;
}

// CHECK: @D.cb. = external constant { [3 x <3 x double>], <3 x half> }
cbuffer D {
  double3 D9[3] : packoffset(c);
  half3 D10;
}
                                
struct S0
{
    double B0;
    float3 B1;
    float B2;
    double3 B3;
    half B4;
    double2 B5;
    float B6;
    half3 B7;
    half3 B8;
};

struct S1
{
    float A0;
    double A1;
    float A2;
    half A3;
    int16_t A4;
    int64_t A5;
    int A6;
};

struct S2
{
    double B0;
    float3 B1;
    float B2;
    double3 B3;
    half B4;
    double2 B5;
    float B6;
    half3 B7;
    half3 B8;
};

struct S3
{
    S1 C0;
    float C1[1];
    S2 C2[2];
    half C3;
};           

// CHECK: @E.cb. = external constant { [16 x i8], half, [2 x i8], i32, double, %struct.S3, [206 x i8], %struct.S0 }
cbuffer E {
  int E0 : packoffset(c1.y);
  S0  E1 : packoffset(c31);
  half E2 : packoffset(c1);
  S3 E3 : packoffset(c2);
  double E4 : packoffset(c1.z);
}

float foo() {
// CHECK: %[[a:.+]] = load float, ptr getelementptr inbounds ({ [8 x i8], double, [8 x i8], float, float, half, i16, [4 x i8], i64, i32 }, ptr @A.cb., i32 0, i32 3), align 4
// CHECK: %[[B2:.+]] = load float, ptr getelementptr inbounds ({ [24 x i8], float, [4 x i8], double, [8 x i8], <3 x float>, [4 x i8], <3 x double>, half, [6 x i8], <2 x double>, float, <3 x half>, <3 x half> }, ptr @B.cb., i32 0, i32 1), align 4
// CHECK: %[[C6:.+]] = load float, ptr getelementptr inbounds ({ [2 x double], [8 x i8], [3 x <3 x float>], float, [3 x double], half, [6 x i8], [1 x <2 x double>], float, [12 x i8], [2 x <3 x half>], <3 x half> }, ptr @C.cb., i32 0, i32 8), align 4
// CHECK: %[[D10:.+]] = load <3 x half>, ptr getelementptr inbounds ({ [3 x <3 x double>], <3 x half> }, ptr @D.cb., i32 0, i32 1), align 8
// CHECK: %[[E0:.+]] = load i32, ptr getelementptr inbounds ({ [16 x i8], half, [2 x i8], i32, double, %struct.S3, [206 x i8], %struct.S0 }, ptr @E.cb., i32 0, i32 3), align 4
  return a + B2 + C6*D10.z + E0;
}
