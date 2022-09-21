// RUN: %clang_cc1 -triple armv8-none-linux-gnueabi -target-feature +crc \
// RUN:  -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s
// RUN: %clang_cc1 -verify -emit-llvm -triple armv7-none-linux-gnueabi %s

int crc32b(int a, char b)
{
// expected-error@+1 {{'__builtin_arm_crc32b' needs target feature crc}}
        return __builtin_arm_crc32b(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i8 %b to i32
// CHECK: call i32 @llvm.arm.crc32b(i32 %a, i32 [[T0]])
}

int crc32cb(int a, char b)
{
        return __builtin_arm_crc32cb(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i8 %b to i32
// CHECK: call i32 @llvm.arm.crc32cb(i32 %a, i32 [[T0]])
}

int crc32h(int a, short b)
{
        return __builtin_arm_crc32h(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i16 %b to i32
// CHECK: call i32 @llvm.arm.crc32h(i32 %a, i32 [[T0]])
}

int crc32ch(int a, short b)
{
        return __builtin_arm_crc32ch(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i16 %b to i32
// CHECK: call i32 @llvm.arm.crc32ch(i32 %a, i32 [[T0]])
}

int crc32w(int a, int b)
{
        return __builtin_arm_crc32w(a,b);
// CHECK: call i32 @llvm.arm.crc32w(i32 %a, i32 %b)
}

int crc32cw(int a, int b)
{
        return __builtin_arm_crc32cw(a,b);
// CHECK: call i32 @llvm.arm.crc32cw(i32 %a, i32 %b)
}

int crc32d(int a, long long b)
{
        return __builtin_arm_crc32d(a,b);
// CHECK: [[T0:%[0-9]+]] = trunc i64 %b to i32
// CHECK: [[T1:%[0-9]+]] = lshr i64 %b, 32
// CHECK: [[T2:%[0-9]+]] = trunc i64 [[T1]] to i32
// CHECK: [[T3:%[0-9]+]] = call i32 @llvm.arm.crc32w(i32 %a, i32 [[T0]])
// CHECK: call i32 @llvm.arm.crc32w(i32 [[T3]], i32 [[T2]])
}

int crc32cd(int a, long long b)
{
        return __builtin_arm_crc32cd(a,b);
// CHECK: [[T0:%[0-9]+]] = trunc i64 %b to i32
// CHECK: [[T1:%[0-9]+]] = lshr i64 %b, 32
// CHECK: [[T2:%[0-9]+]] = trunc i64 [[T1]] to i32
// CHECK: [[T3:%[0-9]+]] = call i32 @llvm.arm.crc32cw(i32 %a, i32 [[T0]])
// CHECK: call i32 @llvm.arm.crc32cw(i32 [[T3]], i32 [[T2]])
}
