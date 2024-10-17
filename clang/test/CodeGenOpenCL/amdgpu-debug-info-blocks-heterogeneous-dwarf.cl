// REQUIRES: amdgpu-registered-target
// RUN: %clang -Xclang -cl-std=CL2.0 -emit-llvm -g -gheterogeneous-dwarf=diexpr -O0 -S -nogpulib -target amdgcn-amd-amdhsa -mcpu=gfx1010 -o - %s | FileCheck %s
// RUN: %clang -Xclang -cl-std=CL2.0 -emit-llvm -g -gheterogeneous-dwarf=diexpr -O0 -S -nogpulib -target amdgcn-amd-amdhsa-opencl -mcpu=gfx1010 -o - %s | FileCheck %s

// FIXME: Currently just testing that we don't crash; test for the absense
// of meaningful debug information for the block is to identify this test
// to update/replace when this is implemented.

// CHECK-NOT: call void @llvm.dbg.{{.*}}%block.capture.addr

void fn(__global uint* res);
__kernel void kern(__global uint* res) {
  void (^kernelBlock)(void) = ^{ fn(res); };
}
