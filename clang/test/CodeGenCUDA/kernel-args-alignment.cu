// New CUDA kernel launch sequence does not require explicit specification of
// size/offset for each argument, so only the old way is tested.
//
// RUN: %clang_cc1 --std=c++11 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:    -target-sdk-version=8.0 -o - %s \
// RUN:  | FileCheck -check-prefixes=HOST-OLD,HOST-OLD-NV,CHECK %s

// RUN: %clang_cc1 --std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda \
// RUN:   -emit-llvm -o - %s | FileCheck -check-prefixes=DEVICE,DEVICE-NV,CHECK %s

// RUN: %clang_cc1 --std=c++11 -triple x86_64-unknown-linux-gnu -x hip \
// RUN:  -aux-triple amdgcn-amd-amdhsa -emit-llvm -o - %s | \
// RUN:  FileCheck -check-prefixes=HOST-OLD,HOST-OLD-AMD,CHECK %s

// RUN: %clang_cc1 --std=c++11 -fcuda-is-device -triple amdgcn-amd-amdhsa \
// RUN:  -x hip -emit-llvm -o - %s | FileCheck -check-prefixes=DEVICE,DEVICE-AMD,CHECK %s

#include "Inputs/cuda.h"

struct U {
  short x;
} __attribute__((packed));

struct S {
  int *ptr;
  char a;
  U u;
};

// Clang should generate a packed LLVM struct for S (denoted by the <>s),
// otherwise this test isn't interesting.
// CHECK: %struct.S = type <{ i32*, i8, %struct.U, [5 x i8] }>

static_assert(alignof(S) == 8, "Unexpected alignment.");

// HOST-LABEL: @_Z6kernelc1SPi
// For NVPTX backend, marshalled kernel args should be:
//   1. offset 0, width 1
//   2. offset 8 (because alignof(S) == 8), width 16
//   3. offset 24, width 8
// HOST-NV-OLD: call i32 @cudaSetupArgument({{[^,]*}}, i64 1, i64 0)
// HOST-NV-OLD: call i32 @cudaSetupArgument({{[^,]*}}, i64 16, i64 8)
// HOST-NV-OLD: call i32 @cudaSetupArgument({{[^,]*}}, i64 8, i64 24)
// AMDGPU backend assumes struct type kernel arguments are passed directly,
// not byval. It lays out kernel arguments by size and alignment in IR.
// Packed struct type in IR always has ABI alignment of 1.
// For AMDGPU backend, marshalled kernel args should be:
//   1. offset 0, width 1
//   2. offset 1 (because ABI alignment of S is 1), width 16
//   3. offset 24, width 8
// HOST-AMD: call i32 @hipSetupArgument({{[^,]*}}, i64 1, i64 0)
// HOST-AMD: call i32 @hipSetupArgument({{[^,]*}}, i64 16, i64 1)
// HOST-AMD: call i32 @hipSetupArgument({{[^,]*}}, i64 8, i64 24)

// DEVICE-LABEL: @_Z6kernelc1SPi
// DEVICE-NV-SAME: i8{{[^,]*}}, %struct.S* byval(%struct.S) align 8{{[^,]*}}, i32*
// DEVICE-AMD-SAME: i8{{[^,]*}}, %struct.S{{[^,*]*}}, i32*
__global__ void kernel(char a, S s, int *b) {}
