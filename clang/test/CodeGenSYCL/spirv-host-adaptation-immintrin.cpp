/// Check that use of Clang's x86 intrinsics in SPIR-V device code is diagnosed
/// at compile time when the auxiliary host target is x86-64 MSVC, including
/// when sse/sse2 are enabled explicitly for the device.

/// The intrinsics require host target features, so they are host-only.
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -ffreestanding -emit-llvm -verify=sse -o - %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -ffreestanding -target-feature +sse -target-feature +sse2 \
// RUN:   -emit-llvm -verify=sse -o - %s

/// Intrinsics that lower to x86 builtins are rejected for the device.
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -ffreestanding -DX86_BUILTIN -emit-llvm -verify=x86-builtin \
// RUN:   -o - %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -ffreestanding -target-feature +sse -target-feature +sse2 \
// RUN:   -DX86_BUILTIN -emit-llvm -verify=x86-builtin -o - %s

#include <immintrin.h>

#ifndef X86_BUILTIN
[[clang::sycl_external]] float add(float x, float y) {
  // sse-error@+1 {{SYCL device code cannot call '_mm_set1_ps', which requires the 'sse' host feature}}
  __m128 a = _mm_set1_ps(x);
  // sse-error@+1 {{SYCL device code cannot call '_mm_set1_ps', which requires the 'sse' host feature}}
  __m128 b = _mm_set1_ps(y);
  // sse-error@+1 {{SYCL device code cannot call '_mm_add_ps', which requires the 'sse' host feature}}
  __m128 c = _mm_add_ps(a, b);
  return c[0];
}
#else
// The diagnostic is reported in the intrinsic header, not here.
// x86-builtin-error@*:* {{builtin is not supported on this target}}
[[clang::sycl_external]] __m128i madd(__m128i a, __m128i b) {
  // x86-builtin-note@+2 {{called by 'madd'}}
  // x86-builtin-error@+1 {{SYCL device code cannot call '_mm_madd_epi16', which requires the 'sse2' host feature}}
  return _mm_madd_epi16(a, b);
}
#endif
