// REQUIRES: amdgpu-registered-target, spirv-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:    -Wno-format-security -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:    -mprintf-kind=buffered -Wno-format-security -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device \
// RUN:    -Wno-format-security -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device \
// RUN:    -mprintf-kind=buffered -Wno-format-security -verify -emit-llvm-only %s

// The %s specifiers can only be found in a constant format string, so a
// non-constant format string is rejected if any argument is a pointer.

#define __device__ __attribute__((device))
extern "C" __device__ int printf(const char *format, ...);

__device__ void NonConstFmtNoPtrArgs(const char *fmt) {
  printf(fmt);
  printf(fmt, 1, 2.0, 'c');
}

__device__ void ConstFmtPtrArgs(const char *s, int *p) {
  printf("%s %p\n", s, p);
}

// CodeGen stops emitting functions after the first error, so all error cases
// share the last function.
__device__ void NonConstFmtPtrArgs(const char *fmt, const char *s, int *p) {
  printf(fmt, s); // expected-error {{AMDGPU printf requires a constant format string when pointer arguments are passed}}
  printf(fmt, 1, p); // expected-error {{AMDGPU printf requires a constant format string when pointer arguments are passed}}
  printf(fmt, nullptr); // expected-error {{AMDGPU printf requires a constant format string when pointer arguments are passed}}
}
