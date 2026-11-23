// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s

// Regression test for issue #239858

// CHECK: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", i32, 1, 0, 1) }

struct A {
  RWBuffer<int> Buf;

  int get(uint Index) const {
    return this.Buf[Index] + this.Buf.Load(Index);;
  }
};

// Global variables for resources in `a` and `array`
//
// CHECK-DAG: @a.Buf = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK-DAG: @[[aBufStr:.*]] = private unnamed_addr constant [6 x i8] c"a.Buf\00", align 1
// CHECK-DAG: @array.0.Buf = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK-DAG: @[[array0BufStr:.*]] = private unnamed_addr constant [12 x i8] c"array.0.Buf\00", align 1
// CHECK-DAG: @array.1.Buf = internal global %"class.hlsl::RWBuffer" poison, align 4
// CHECK-DAG: @[[array1BufStr:.*]] = private unnamed_addr constant [12 x i8] c"array.1.Buf\00", align 1

A a : register(u5);
A array[2];

[numthreads(8, 1, 1)]
void main(uint3 ID : SV_GroupThreadID) {
  a.Buf[ID.x] = a.get(ID.x) + array[1].get(ID.x);
}
