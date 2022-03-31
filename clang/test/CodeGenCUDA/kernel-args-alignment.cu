// New CUDA kernel launch sequence does not require explicit specification of
// size/offset for each argument, so only the old way is tested.
//
// RUN: %clang_cc1 --std=c++11 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:    -target-sdk-version=8.0 -o - %s \
// RUN:  | FileCheck -check-prefixes=HOST-OLD,CHECK %s

// RUN: %clang_cc1 --std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda \
// RUN:   -emit-llvm -o - %s | FileCheck -check-prefixes=DEVICE,CHECK %s

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
// Marshalled kernel args should be:
//   1. offset 0, width 1
//   2. offset 8 (because alignof(S) == 8), width 16
//   3. offset 24, width 8
// HOST-OLD: call i32 @cudaSetupArgument({{[^,]*}}, i64 1, i64 0)
// HOST-OLD: call i32 @cudaSetupArgument({{[^,]*}}, i64 16, i64 8)
// HOST-OLD: call i32 @cudaSetupArgument({{[^,]*}}, i64 8, i64 24)

// DEVICE-LABEL: @_Z6kernelc1SPi
// DEVICE-SAME: i8{{[^,]*}}, %struct.S* noundef byval(%struct.S) align 8{{[^,]*}}, i32*
__global__ void kernel(char a, S s, int *b) {}
