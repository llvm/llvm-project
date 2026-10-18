/// Check that the x86 intrinsic headers parse and compile for a SPIR-V device
/// when the auxiliary host target is x86-64 MSVC, which is what the sse/sse2
/// device features derived from that host are for.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -ffreestanding -emit-llvm -o - %s | FileCheck %s

/// With sse/sse2 disabled, the always_inline intrinsics cannot be inlined into
/// device code; this is the failure the derived features prevent.
/// Codegen stops at the first failing function, so only add() is diagnosed.
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -ffreestanding -target-feature -sse -target-feature -sse2 \
// RUN:   -emit-llvm -verify=no-sse -o - %s

/// Intrinsics that lower to x86 builtins are rejected for the device.
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -ffreestanding -DX86_BUILTIN -emit-llvm -verify=x86-builtin \
// RUN:   -o - %s

#include <immintrin.h>

// CHECK-LABEL: define {{.*}}spir_func noundef float @_Z3addff
[[clang::sycl_external]] float add(float x, float y) {
  // no-sse-error@+1 {{always_inline function '_mm_set1_ps' requires target feature 'sse', but would be inlined into function 'add' that is compiled without support for 'sse'}}
  __m128 a = _mm_set1_ps(x);
  // no-sse-error@+1 {{always_inline function '_mm_set1_ps' requires target feature 'sse', but would be inlined into function 'add' that is compiled without support for 'sse'}}
  __m128 b = _mm_set1_ps(y);
  // CHECK: fadd <4 x float>
  // no-sse-error@+1 {{always_inline function '_mm_add_ps' requires target feature 'sse', but would be inlined into function 'add' that is compiled without support for 'sse'}}
  __m128 c = _mm_add_ps(a, b);
  return c[0];
}

// CHECK-LABEL: define {{.*}}spir_func void @_Z4sqrtPd
[[clang::sycl_external]] void sqrt(double *p) {
  __m128d a = _mm_set1_pd(*p);
  // CHECK: call {{.*}}<2 x double> @llvm.sqrt.v2f64
  __m128d b = _mm_sqrt_pd(a);
  _mm_storeu_pd(p, b);
}

#ifdef X86_BUILTIN
// The diagnostic is reported in the intrinsic header, not here.
// x86-builtin-error@*:* {{cannot compile this builtin function yet}}
[[clang::sycl_external]] __m128i madd(__m128i a, __m128i b) {
  return _mm_madd_epi16(a, b);
}
#endif
