// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu \
// RUN:   -S -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

int crc32b(int a, char b)
{
        return __builtin_arm_crc32b(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i8 %b to i32
// CHECK: call i32 @llvm.aarch64.crc32b(i32 %a, i32 [[T0]])
}

int crc32cb(int a, char b)
{
        return __builtin_arm_crc32cb(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i8 %b to i32
// CHECK: call i32 @llvm.aarch64.crc32cb(i32 %a, i32 [[T0]])
}

int crc32h(int a, short b)
{
        return __builtin_arm_crc32h(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i16 %b to i32
// CHECK: call i32 @llvm.aarch64.crc32h(i32 %a, i32 [[T0]])
}

int crc32ch(int a, short b)
{
        return __builtin_arm_crc32ch(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i16 %b to i32
// CHECK: call i32 @llvm.aarch64.crc32ch(i32 %a, i32 [[T0]])
}

int crc32w(int a, int b)
{
        return __builtin_arm_crc32w(a,b);
// CHECK: call i32 @llvm.aarch64.crc32w(i32 %a, i32 %b)
}

int crc32cw(int a, int b)
{
        return __builtin_arm_crc32cw(a,b);
// CHECK: call i32 @llvm.aarch64.crc32cw(i32 %a, i32 %b)
}

int crc32d(int a, long b)
{
        return __builtin_arm_crc32d(a,b);
// CHECK: call i32 @llvm.aarch64.crc32x(i32 %a, i64 %b)
}

int crc32cd(int a, long b)
{
        return __builtin_arm_crc32cd(a,b);
// CHECK: call i32 @llvm.aarch64.crc32cx(i32 %a, i64 %b)
}
