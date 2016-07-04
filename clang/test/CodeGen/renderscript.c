// RUN: %clang_cc1 %s -triple=renderscript32-none-linux-gnueabi -emit-llvm -o - -Werror | FileCheck %s -check-prefix=CHECK-RS32
// RUN: %clang_cc1 %s -triple=renderscript64-none-linux-android -emit-llvm -o - -Werror | FileCheck %s -check-prefix=CHECK-RS64
// RUN: %clang_cc1 %s -triple=armv7-none-linux-gnueabi -emit-llvm -o - -Werror | FileCheck %s -check-prefix=CHECK-ARM

// Ensure that the bitcode has the correct triple
// CHECK-RS32: target triple = "armv7-none-linux-gnueabi"
// CHECK-RS64: target triple = "aarch64-none-linux-android"
// CHECK-ARM: target triple = "armv7-none-linux-gnueabi"

// Ensure that long data type has 8-byte size and alignment in RenderScript
#ifdef __RENDERSCRIPT__
#define LONG_WIDTH_AND_ALIGN 8
#else
#define LONG_WIDTH_AND_ALIGN 4
#endif

_Static_assert(sizeof(long) == LONG_WIDTH_AND_ALIGN, "sizeof long is wrong");
_Static_assert(_Alignof(long) == LONG_WIDTH_AND_ALIGN, "sizeof long is wrong");

// CHECK-RS32: i64 @test_long(i64 %v)
// CHECK-RS64: i64 @test_long(i64 %v)
// CHECK-ARM: i32 @test_long(i32 %v)
long test_long(long v) {
  return v + 1;
}
