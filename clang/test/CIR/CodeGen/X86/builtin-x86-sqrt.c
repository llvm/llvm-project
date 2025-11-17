// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -o - %s 2>&1 | FileCheck %s

// Minimal stand-in for the SSE vector type.
typedef float __m128 __attribute__((__vector_size__(16)));

// Declare the builtin explicitly so we don't need headers.
__m128 __builtin_ia32_sqrtps(__m128);

__m128 test_sqrtps(__m128 a) {
  return __builtin_ia32_sqrtps(a);
}

// CHECK: error: ClangIR code gen Not Yet Implemented: CIR lowering for x86 sqrt builtins is not implemented yet
// CHECK: error: ClangIR code gen Not Yet Implemented: unimplemented builtin call: __builtin_ia32_sqrtps
