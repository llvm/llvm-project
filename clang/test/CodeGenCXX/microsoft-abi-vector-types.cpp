// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=i686-pc-windows-msvc \
// RUN:   | FileCheck --check-prefixes=X86 %s
// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc \
// RUN:   | FileCheck --check-prefixes=X86 %s
// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=aarch64-pc-windows-msvc \
// RUN:   | FileCheck --check-prefixes=AARCH64 %s
// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=i686-pc-windows-msvc \
// RUN:   -fclang-abi-compat=21 | FileCheck --check-prefixes=X86-CLANG21 %s
// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc \
// RUN:   -fclang-abi-compat=21 | FileCheck --check-prefixes=X86-CLANG21 %s
// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=aarch64-pc-windows-msvc \
// RUN:   -fclang-abi-compat=21 | FileCheck --check-prefixes=AARCH64-CLANG21 %s

// To match the MSVC ABI, vector types must be returned indirectly from member
// functions on x86 and x86-64 (as long as they do not use the vectorcall
// calling convention), but must be returned directly everywhere else.

#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>

struct Foo {
  __m128 method_m128();
  __m128 __vectorcall vectorcall_method_m128();
};

__m128 Foo::method_m128() {
  return __m128{};
// GH104
// X86: store <4 x float>
// X86: ret void
// X86-CLANG21: ret <4 x float>
}

__m128 __vectorcall Foo::vectorcall_method_m128() {
  return __m128{};
// X86: ret <4 x float>
}

__m128 func_m128() {
  return __m128{};
// X86: ret <4 x float>
}

__m128 __vectorcall vectorcall_func_m128() {
  return __m128{};
// X86: ret <4 x float>
}
#endif

#ifdef __aarch64__
#include <arm_neon.h>

struct Foo {
  float32x4_t method_f32x4();
  float32x4_t __vectorcall vectorcall_method_f32x4();
};

float32x4_t Foo::method_f32x4() {
  return float32x4_t{};
// AARCH64: ret <4 x float>
// AARCH64-CLANG21: ret <4 x float>
}

float32x4_t __vectorcall Foo::vectorcall_method_f32x4() {
  return float32x4_t{};
// AARCH64: ret <4 x float>
}

float32x4_t func_f32x4() {
  return float32x4_t{};
// AARCH64: ret <4 x float>
}

float32x4_t __vectorcall vectorcall_func_f32x4() {
  return float32x4_t{};
// AARCH64: ret <4 x float>
}
#endif
