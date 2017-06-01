// RUN: %clang_cc1 -Wall -Werror -triple thumbv8-linux-gnueabi -fno-signed-char -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -Wall -Werror -triple arm64-apple-ios7.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARM64

bool b;

// CHECK-LABEL: @_Z10test_ldrexv()
// CHECK: call i32 @llvm.arm.ldrex.p0i8(i8* @b)

// CHECK-ARM64-LABEL: @_Z10test_ldrexv()
// CHECK-ARM64: call i64 @llvm.aarch64.ldxr.p0i8(i8* @b)

void test_ldrex() {
  b = __builtin_arm_ldrex(&b);
}

// CHECK-LABEL: @_Z10tset_strexv()
// CHECK: %{{.*}} = call i32 @llvm.arm.strex.p0i8(i32 1, i8* @b)

// CHECK-ARM64-LABEL: @_Z10tset_strexv()
// CHECK-ARM64: %{{.*}} = call i32 @llvm.aarch64.stxr.p0i8(i64 1, i8* @b)

void tset_strex() {
  __builtin_arm_strex(true, &b);
}
