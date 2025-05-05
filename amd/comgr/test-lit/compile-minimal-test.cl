// COM: Run Comgr binary to compile OpenCL source into LLVM IR Bitcode,
// COM: and, then generate an executable
// RUN: compile-minimal-test %s %t.bin

// COM: Dissasemble
// RUN: llvm-objdump -d %t.bin | FileCheck %s
// CHECK: <add>:
// CHECK: s_endpgm

void kernel add(__global float *A, __global float *B, __global float *C) {
    *C = *A + *B;
}
