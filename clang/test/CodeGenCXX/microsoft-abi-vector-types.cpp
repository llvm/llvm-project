// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=i686-pc-windows-msvc \
// RUN:   | FileCheck --check-prefixes=CHECK,X86 %s
// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc \
// RUN:   | FileCheck --check-prefixes=CHECK,X86 %s
// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=aarch64-pc-windows-msvc \
// RUN:   | FileCheck --check-prefixes=CHECK,AARCH64 %s
// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=i686-pc-windows-msvc \
// RUN:   -fclang-abi-compat=21 | FileCheck --check-prefixes=CHECK,CLANG21 %s
// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc \
// RUN:   -fclang-abi-compat=21 | FileCheck --check-prefixes=CHECK,CLANG21 %s
// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=aarch64-pc-windows-msvc \
// RUN:   -fclang-abi-compat=21 | FileCheck --check-prefixes=CHECK,CLANG21 %s

// To match the MSVC ABI, vector types must be returned indirectly from member
// functions on x86 and x86-64 (as long as they do not use the vectorcall
// calling convention), but must be returned directly everywhere else.

#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#define VECTOR_TYPE __m128
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#define VECTOR_TYPE float32x4_t
#endif

struct Foo {
  VECTOR_TYPE method_ret_vec();
  VECTOR_TYPE __vectorcall vectorcall_method_ret_vec();
};

VECTOR_TYPE Foo::method_ret_vec() {
  return VECTOR_TYPE{};
// GH104
// X86: store <4 x float>
// X86: ret void
// AARCH64: ret <4 x float>
// CLANG21: ret <4 x float>
}

VECTOR_TYPE __vectorcall Foo::vectorcall_method_ret_vec() {
  return VECTOR_TYPE{};
// CHECK: ret <4 x float>
}

VECTOR_TYPE func_ret_vec() {
  return VECTOR_TYPE{};
// CHECK: ret <4 x float>
}

VECTOR_TYPE __vectorcall vectorcall_func_ret_vec() {
  return VECTOR_TYPE{};
// CHECK: ret <4 x float>
}
