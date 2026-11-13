/// Check that use of Clang's x86 intrinsics in SPIR-V device code is diagnosed
/// at compile time when the auxiliary host target is x86-64 MSVC.

/// The always_inline intrinsics require sse/sse2, which are not enabled for the
/// device.
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -ffreestanding -emit-llvm -verify=no-sse -o - %s

/// Intrinsics that lower to x86 builtins are rejected for the device.
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -ffreestanding -DX86_BUILTIN -emit-llvm -verify=x86-builtin \
// RUN:   -o - %s

#include <immintrin.h>

#ifndef X86_BUILTIN
[[clang::sycl_external]] float add(float x, float y) {
  // no-sse-error@+1 {{always_inline function '_mm_set1_ps' requires target feature 'sse', but would be inlined into function 'add' that is compiled without support for 'sse'}}
  __m128 a = _mm_set1_ps(x);
  // no-sse-error@+1 {{always_inline function '_mm_set1_ps' requires target feature 'sse', but would be inlined into function 'add' that is compiled without support for 'sse'}}
  __m128 b = _mm_set1_ps(y);
  // no-sse-error@+1 {{always_inline function '_mm_add_ps' requires target feature 'sse', but would be inlined into function 'add' that is compiled without support for 'sse'}}
  __m128 c = _mm_add_ps(a, b);
  return c[0];
}
#else
// The diagnostic is reported in the intrinsic header, not here.
// x86-builtin-error@*:* {{builtin is not supported on this target}}
[[clang::sycl_external]] __m128i madd(__m128i a, __m128i b) {
  // x86-builtin-note@+2 {{called by 'madd'}}
  // x86-builtin-error@+1 {{always_inline function '_mm_madd_epi16' requires target feature 'sse2', but would be inlined into function 'madd' that is compiled without support for 'sse2'}}
  return _mm_madd_epi16(a, b);
}
#endif
