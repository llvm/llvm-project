// RUN: %clang_cc1 -Wall -Werror -triple thumbv8-linux-gnueabi -fno-signed-char -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -Wall -Werror -triple arm64-apple-ios7.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ARM64

bool b;

// CHECK-LABEL: @_Z10test_ldrexv()
// CHECK: call i32 @llvm.arm.ldrex.p0(ptr elementtype(i8) @b)

// CHECK-ARM64-LABEL: @_Z10test_ldrexv()
// CHECK-ARM64: call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i8) @b)

void test_ldrex() {
  b = __builtin_arm_ldrex(&b);
}

// CHECK-LABEL: @_Z10tset_strexv()
// CHECK: %{{.*}} = call i32 @llvm.arm.strex.p0(i32 1, ptr elementtype(i8) @b)

// CHECK-ARM64-LABEL: @_Z10tset_strexv()
// CHECK-ARM64: %{{.*}} = call i32 @llvm.aarch64.stxr.p0(i64 1, ptr elementtype(i8) @b)

void tset_strex() {
  __builtin_arm_strex(true, &b);
}

#ifdef __arm__
// ARM exclusive atomic builtins

long long c;

// CHECK-LABEL: @_Z11test_ldrexdv()
// CHECK: [[STRUCTRES:%.*]] = call { i32, i32 } @llvm.arm.ldrexd(ptr @c)
// CHECK: [[RESHI:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 1
// CHECK: [[RESLO:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 0
// CHECK: [[RESHI64:%.*]] = zext i32 [[RESHI]] to i64
// CHECK: [[RESLO64:%.*]] = zext i32 [[RESLO]] to i64
// CHECK: [[RESHIHI:%.*]] = shl nuw i64 [[RESHI64]], 32
// CHECK: [[INTRES:%.*]] = or i64 [[RESHIHI]], [[RESLO64]]
// CHECK: store i64 [[INTRES]], ptr @c, align 8

void test_ldrexd() {
  c = __builtin_arm_ldrexd(&c);
}

// CHECK-LABEL: @_Z11tset_strexdv()
// CHECK: store i64 42, ptr [[TMP:%.*]], align 8
// CHECK: [[LOHI:%.*]] = load { i32, i32 }, ptr [[TMP]]
// CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
// CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
// CHECK: %{{.*}} = call i32 @llvm.arm.strexd(i32 [[LO]], i32 [[HI]], ptr @c)

void tset_strexd() {
  __builtin_arm_strexd(42, &c);
}

#endif
