/// The MSVC intrinsic headers declare most _mm_* intrinsics as plain extern
/// functions, which Clang does not treat as builtins. Device code that calls
/// them is not diagnosed and leaves unresolved external references.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -fms-extensions -emit-llvm -verify -o - %s | FileCheck %s

// expected-no-diagnostics

typedef union __declspec(align(16)) __m128 {
  float m128_f32[4];
} __m128;

extern "C" {
extern __m128 __cdecl _mm_set_ps1(float);
extern __m128 __cdecl _mm_add_ps(__m128, __m128);
}
#define _mm_set1_ps _mm_set_ps1

// CHECK: call spir_func {{.*}}@_mm_set_ps1(
// CHECK: call spir_func {{.*}}@_mm_add_ps(
[[clang::sycl_external]] float add(float x) {
  __m128 a = _mm_set1_ps(x);
  return _mm_add_ps(a, a).m128_f32[0];
}

// CHECK: declare spir_func {{.*}}@_mm_set_ps1(
// CHECK: declare spir_func {{.*}}@_mm_add_ps(
