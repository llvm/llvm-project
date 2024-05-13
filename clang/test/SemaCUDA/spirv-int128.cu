// RUN: %clang_cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -verify -fsyntax-only %s

#define __device__ __attribute__((device))

__int128 h_glb;

__device__ __int128 d_unused;

// expected-note@+1 {{'d_glb' defined here}}
__device__ __int128 d_glb;

__device__ __int128 bar() {
  // expected-error@+1 {{'d_glb' requires 128 bit size '__int128' type support, but target 'spirv64' does not support it}}
  return d_glb;
}
