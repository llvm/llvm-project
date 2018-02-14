// RUN: %clang_cc1 -emit-llvm -triple "spir-unknown-unknown" -O0 -cl-std=CL2.0 -o - %s | FileCheck -check-prefixes=CHECK,SPIR %s
// RUN: %clang_cc1 -emit-llvm -triple amdgcn-amd-amdhsa -O0 -cl-std=CL2.0 -o - %s | FileCheck -check-prefixes=CHECK,AMD %s

constant int sz0 = 5;
// SPIR: @sz0 = addrspace(2) constant i32 5
// AMD: @sz0 = addrspace(4) constant i32 5
const global int sz1 = 16;
// CHECK: @sz1 = addrspace(1) constant i32 16
const constant int sz2 = 8;
// SPIR: @sz2 = addrspace(2) constant i32 8
// AMD: @sz2 = addrspace(4) constant i32 8
// CHECK: @testvla.vla2 = internal addrspace(3) global [8 x i16] undef

kernel void testvla()
{
  int vla0[sz0];
// SPIR: %vla0 = alloca [5 x i32]
// SPIR-NOT: %vla0 = alloca [5 x i32]{{.*}}addrspace
// GIZ: %vla0 = alloca [5 x i32]{{.*}}addrspace(5)
  char vla1[sz1];
// SPIR: %vla1 = alloca [16 x i8]
// SPIR-NOT: %vla1 = alloca [16 x i8]{{.*}}addrspace
// GIZ: %vla1 = alloca [16 x i8]{{.*}}addrspace(5)
  local short vla2[sz2];
}
