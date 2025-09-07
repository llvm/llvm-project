// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=i686-pc-windows-msvc | FileCheck %s
// RUN: %clang_cc1 -ffreestanding -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc | FileCheck %s

// To match the MSVC ABI, vector types must be returned indirectly from member
// functions (as long as they do not use the vectorcall calling convention),
// but must be returned directly everywhere else.

#include <xmmintrin.h>

struct Foo {
  __m128 method_m128();
  __m128 __vectorcall vectorcall_method_m128();
};

__m128 Foo::method_m128() {
  return __m128{};
// GH104
// CHECK: store <4 x float>
// CHECK: ret void
}

__m128 __vectorcall Foo::vectorcall_method_m128() {
  return __m128{};
// CHECK: ret <4 x float>
}

__m128 func_m128() {
  return __m128{};
// CHECK: ret <4 x float>
}

__m128 __vectorcall vectorcall_func_m128() {
  return __m128{};
// CHECK: ret <4 x float>
}

