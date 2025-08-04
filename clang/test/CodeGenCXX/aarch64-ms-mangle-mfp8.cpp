// RUN: %clang_cc1 -triple aarch64-windows-msvc -emit-llvm -o - %s | FileCheck %s

typedef __mfp8 mf8;
typedef __attribute__((neon_vector_type(8))) __mfp8 mf8x8_t;
typedef __attribute__((neon_vector_type(16))) __mfp8 mf8x16_t;

// CHECK: "?f@@YAXU__mfp8@__clang@@@Z"
void f(mf8 v) {}

// CHECK: "?f@@YAXT?$__vector@U__mfp8@__clang@@$07@__clang@@@Z"
void f(mf8x8_t v) {}

// CHECK: "?f@@YAXT?$__vector@U__mfp8@__clang@@$0BA@@__clang@@@Z"
void f(mf8x16_t v) {}
