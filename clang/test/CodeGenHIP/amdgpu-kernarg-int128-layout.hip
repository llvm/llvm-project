// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -target-cpu gfx900 -S -o - %s | FileCheck %s

// Check that the kernarg segment layout the backend emits agrees with the
// C-level layout Clang guarantees for __int128 (AMDGPUTargetInfo leaves
// Int128Align at its 128-bit default). The static_asserts pin the frontend
// side; the .offset/.size metadata entries pin the backend side. Before the
// data layout gave i128 a default ABI alignment of 16 the backend placed
// i128 kernel arguments at the next 8-byte slot, so host code computing
// argument offsets with offsetof disagreed with the device-side layout.

#define __global__ __attribute__((global))

struct S {
  char c;
  __int128 x;
};

static_assert(__alignof__(__int128) == 16);
static_assert(__builtin_offsetof(S, x) == 16);
static_assert(sizeof(S) == 32);

// CHECK-LABEL: {{^}}scalar_args:
extern "C" __global__ void scalar_args(char c, __int128 x, int *out) {
  *out = (int)(x >> 64) + c;
}

// CHECK-LABEL: {{^}}struct_arg:
extern "C" __global__ void struct_arg(S s, int *out) {
  *out = (int)(s.x >> 64) + s.c;
}

// The metadata block is emitted once at module scope, after all functions.
// CHECK: .amdgpu_metadata

// `x` must be aligned to 16 in the kernarg segment, not packed at 8, and the
// following pointer argument starts after it at 32.
// CHECK:      .name:           c
// CHECK-NEXT: .offset:         0
// CHECK-NEXT: .size:           1
// CHECK:      .name:           x
// CHECK-NEXT: .offset:         16
// CHECK-NEXT: .size:           16
// CHECK:      .name:           out
// CHECK-NEXT: .offset:         32
// CHECK-NEXT: .size:           8
// CHECK:      .name:           scalar_args

// The by-value struct occupies one 32-byte slot: field `x` lives at byte 16
// of it, matching offsetof(S, x) above.
// CHECK:      .offset:         0
// CHECK-NEXT: .size:           32
// CHECK:      .name:           out
// CHECK-NEXT: .offset:         32
// CHECK-NEXT: .size:           8
// CHECK:      .name:           struct_arg
