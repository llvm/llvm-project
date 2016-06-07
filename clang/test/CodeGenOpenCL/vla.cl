// RUN: %clang_cc1 -emit-llvm -triple "spir-unknown-unknown" -O0 -cl-std=CL2.0 -o - %s | FileCheck %s

constant int sz0 = 5;
// CHECK: @sz0 = addrspace(2) constant i32 5
const global int sz1 = 16;
// CHECK: @sz1 = addrspace(1) constant i32 16
const constant int sz2 = 8;
// CHECK: @sz2 = addrspace(2) constant i32 8
// CHECK: @testvla.vla2 = internal addrspace(3) global [8 x i16] undef

kernel void testvla()
{
  int vla0[sz0];
// CHECK: %vla0 = alloca [5 x i32]
  char vla1[sz1];
// CHECK: %vla1 = alloca [16 x i8]
  local short vla2[sz2];
}
