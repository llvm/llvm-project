// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-pc-win32 | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable


half test()
{
   half x = 0.1f;
   x+=2.0f;
   x-=2.0f;
   half y = x + x;
   half z = y * 1.0f;
   return z;
// CHECK: half 0xH3260
}

// CHECK-LABEL: @test_inc(half noundef %x)
// CHECK: [[INC:%.*]] = fadd half %x, 0xH3C00
// CHECK: ret half [[INC]]
half test_inc(half x)
{
  return ++x;
}

__attribute__((overloadable)) int min(int, int);
__attribute__((overloadable)) half min(half, half);
__attribute__((overloadable)) float min(float, float);

__kernel void foo( __global half* buf, __global float* buf2 )
{
    buf[0] = min( buf[0], 1.5h );
// CHECK: half noundef 0xH3E00
    buf[0] = min( buf2[0], 1.5f );
// CHECK: float noundef 1.500000e+00

    const half one = 1.6666;
    buf[1] = min( buf[1], one );
// CHECK: half noundef 0xH3EAB
}

