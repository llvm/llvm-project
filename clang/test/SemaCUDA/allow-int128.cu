// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple nvptx \
// RUN:   -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -verify -fsyntax-only %s

// expected-no-diagnostics
#define __device__ __attribute__((device))

__int128 h_glb;
__device__ __int128 d_unused;
__device__ __int128 d_glb;
__device__ __int128 bar() {
  return d_glb;
}
