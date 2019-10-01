// RUN: %clang_cc1 -triple aarch64-windows -ffreestanding -emit-llvm -O0 \
// RUN: -x c++ -o - %s | FileCheck %s

struct size1 { char str[1]; };
struct size2 { char str[2]; };
struct size7 { char str[4]; };
struct size8 { char str[8]; };
struct size63 { char str[63]; };
struct size64 { char str[64]; };

struct size1 s1;
// CHECK: @"?s1@@3Usize1@@A" = dso_local global %struct.size1 zeroinitializer, align 1

struct size2 s2;
// CHECK: @"?s2@@3Usize2@@A" = dso_local global %struct.size2 zeroinitializer, align 4

struct size7 s7;
// CHECK: @"?s7@@3Usize7@@A" = dso_local global %struct.size7 zeroinitializer, align 4

struct size8 s8;
// CHECK: @"?s8@@3Usize8@@A" = dso_local global %struct.size8 zeroinitializer, align 8

struct size63 s63;
// CHECK: @"?s63@@3Usize63@@A" = dso_local global %struct.size63 zeroinitializer, align 8

struct size64 s64;
// CHECK: @"?s64@@3Usize64@@A" = dso_local global %struct.size64 zeroinitializer, align 16
