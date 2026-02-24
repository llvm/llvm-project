// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple nvptx \
// RUN:   -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -verify -fsyntax-only %s

// Verify that __int256 is allowed in CUDA device code when the host target
// supports it, matching the __int128 behavior (see allow-int128.cu).
// In CUDA mode, the host type system is shared with the device — type support
// diagnostics are deferred and not emitted for CUDA device compilations.

// expected-no-diagnostics

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__int256 h_glb;
__device__ __int256 d_glb;

__device__ __int256 bar() {
  return d_glb;
}

__global__ void kernel() {
  bar();
}
