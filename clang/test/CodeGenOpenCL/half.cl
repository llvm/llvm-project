// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s

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

// CHECK-LABEL: @test_inc(half %x)
// CHECK: [[INC:%.*]] = fadd half %x, 0xH3C00
// CHECK: ret half [[INC]]
half test_inc(half x)
{
  return ++x;
}
