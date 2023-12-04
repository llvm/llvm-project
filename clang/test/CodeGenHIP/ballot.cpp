// REQUIRES: amdgpu-registered-target
// XFAIL: *
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-windows-msvc -target-cpu gfx900 -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-windows-msvc -target-cpu gfx900 -x hip -S -fcuda-is-device -o - %s | FileCheck %s --check-prefix=GFX9

// Unlike OpenCL, HIP depends on the C++ interpration of "unsigned long", which
// is 64 bits long on Linux and 32 bits long on Windows. The return type of the
// ballot intrinsic needs to be a 64 bit integer on both platforms. This test
// cross-compiles to Windows to confirm that the return type is indeed 64 bits
// on Windows.

// FIXME: The Clang declaration of the wave-64 builtin uses "UL" as the return
// type, which is interpreted as a 32-bit unsigned integer on Windows. This
// emits an incorrect LLVM declaration with i32 return type instead of i64. The
// clang declaration needs to be fixed to use "WU" instead.

// CHECK-LABEL: @_Z3fooi
// CHECK: call i64 @llvm.amdgcn.ballot.i64

// GFX9-LABEL: _Z3fooi:
// GFX9: v_cmp_ne_u32_e64

#define __device__ __attribute__((device))

__device__ unsigned long long foo(int p) {
  return __builtin_amdgcn_ballot_w64(p);
}
