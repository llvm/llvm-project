// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu \
// RUN:  -disable-O0-optnone -S -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-windows \
// RUN:  -disable-O0-optnone -S -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s
#include <stdint.h>

uint32_t crc32b(uint32_t a, uint8_t b)
{
        return __builtin_arm_crc32b(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i8 %b to i32
// CHECK: call i32 @llvm.aarch64.crc32b(i32 %a, i32 [[T0]])
}

uint32_t crc32cb(uint32_t a, uint8_t b)
{
        return __builtin_arm_crc32cb(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i8 %b to i32
// CHECK: call i32 @llvm.aarch64.crc32cb(i32 %a, i32 [[T0]])
}

uint32_t crc32h(uint32_t a, uint16_t b)
{
        return __builtin_arm_crc32h(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i16 %b to i32
// CHECK: call i32 @llvm.aarch64.crc32h(i32 %a, i32 [[T0]])
}

uint32_t crc32ch(uint32_t a, uint16_t b)
{
        return __builtin_arm_crc32ch(a,b);
// CHECK: [[T0:%[0-9]+]] = zext i16 %b to i32
// CHECK: call i32 @llvm.aarch64.crc32ch(i32 %a, i32 [[T0]])
}

uint32_t crc32w(uint32_t a, uint32_t b)
{
        return __builtin_arm_crc32w(a,b);
// CHECK: call i32 @llvm.aarch64.crc32w(i32 %a, i32 %b)
}

uint32_t crc32cw(uint32_t a, uint32_t b)
{
        return __builtin_arm_crc32cw(a,b);
// CHECK: call i32 @llvm.aarch64.crc32cw(i32 %a, i32 %b)
}

uint32_t crc32d(uint32_t a, uint64_t b)
{
        return __builtin_arm_crc32d(a,b);
// CHECK: call i32 @llvm.aarch64.crc32x(i32 %a, i64 %b)
}

uint32_t crc32cd(uint32_t a, uint64_t b)
{
        return __builtin_arm_crc32cd(a,b);
// CHECK: call i32 @llvm.aarch64.crc32cx(i32 %a, i64 %b)
}
