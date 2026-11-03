// -emit-llvm prints the device IR of each input before its host IR.
// RUN: cat %s | clang-repl --cuda -Xcc -Xclang -Xcc -emit-llvm | FileCheck %s

extern "C" __global__ void kernel(int *p) { *p = 42; }
// CHECK: target triple = "nvptx64-nvidia-cuda"
// CHECK: define {{.*}}ptx_kernel void @kernel(
// CHECK: define {{.*}}void @__device_stub__kernel(

%quit
