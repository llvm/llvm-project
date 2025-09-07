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

// To match the MSVC ABI, vector types are usually returned directly, but on x86
// and x86-64 they must be returned indirectly from member functions (unless
// they use the vectorcall calling convention and the vector type is > 64 bits).

#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>

#define VECTOR64_TYPE __m64
#define VECTOR128_TYPE __m128

#define VECTORCALL __vectorcall
#endif

#ifdef __aarch64__
#include <arm_neon.h>

// These were chosen such that they lower to the same types that the x86 vector
// types lower to (e.g. int64x1_t and __m64 both lower to <1 x i64>).
#define VECTOR64_TYPE int64x1_t
#define VECTOR128_TYPE float32x4_t

#define VECTORCALL
#endif

struct Foo {
  VECTOR64_TYPE method_ret_vec64();
  VECTOR128_TYPE method_ret_vec128();

  VECTOR64_TYPE VECTORCALL vc_method_ret_vec64();
  VECTOR128_TYPE VECTORCALL vc_method_ret_vec128();
};

VECTOR64_TYPE Foo::method_ret_vec64() {
  return VECTOR64_TYPE{};
// X86: store <1 x i64>
// X86: ret void
// AARCH64: ret <1 x i64>
// CLANG21: ret <1 x i64>
}

VECTOR128_TYPE Foo::method_ret_vec128() {
  return VECTOR128_TYPE{};
// X86: store <4 x float>
// X86: ret void
// AARCH64: ret <4 x float>
// CLANG21: ret <4 x float>
}

VECTOR64_TYPE VECTORCALL Foo::vc_method_ret_vec64() {
  return VECTOR64_TYPE{};
// X86: store <1 x i64>
// X86: ret void
// AARCH64: ret <1 x i64>
// CLANG21: ret <1 x i64>
}

VECTOR128_TYPE VECTORCALL Foo::vc_method_ret_vec128() {
  return VECTOR128_TYPE{};
// CHECK: ret <4 x float>
}

VECTOR64_TYPE func_ret_vec64() {
  return VECTOR64_TYPE{};
// CHECK: ret <1 x i64>
}

VECTOR128_TYPE func_ret_vec128() {
  return VECTOR128_TYPE{};
// CHECK: ret <4 x float>
}

VECTOR64_TYPE VECTORCALL vc_func_ret_vec64() {
  return VECTOR64_TYPE{};
// CHECK: ret <1 x i64>
}

VECTOR128_TYPE VECTORCALL vc_func_ret_vec128() {
  return VECTOR128_TYPE{};
// CHECK: ret <4 x float>
}
