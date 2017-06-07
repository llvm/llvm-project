// RUN: %clang_cc1 -triple armv7a-none-eabi -target-feature +neon -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=AAPCS
// RUN: %clang_cc1 -triple armv7a-none-gnueabi -target-feature +neon -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=AAPCS
// RUN: %clang_cc1 -triple armv7a-none-freebsd -target-feature +neon -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=AAPCS

// RUN: %clang_cc1 -triple armv7a-apple-ios -target-feature +neon -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=DEFAULT
// RUN: %clang_cc1 -triple armv7a-none-android -target-feature +neon -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=DEFAULT
// RUN: %clang_cc1 -triple armv7a-none-androideabi -target-feature +neon -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=DEFAULT

#include <arm_neon.h>
// Neon types have 64-bit alignment
int32x4_t gl_b;
void t3(int32x4_t *src) {
// CHECK: @t3
  gl_b = *src;
// AAPCS: store <4 x i32> {{%.*}}, <4 x i32>* @gl_b, align 8
// DEFAULT: store <4 x i32> {{%.*}}, <4 x i32>* @gl_b, align 16
}
