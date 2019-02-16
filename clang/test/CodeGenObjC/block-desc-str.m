// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -emit-llvm -fobjc-runtime=gnustep-1.7 -fblocks -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -fobjc-runtime=gcc -fblocks -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm -fblocks -o - %s | FileCheck %s

// Test that descriptor symbol names don't include '@'.

// CHECK: @[[STR:.*]] = private unnamed_addr constant [6 x i8] c"v8@?0\00"
// CHECK: @"__block_descriptor_40_8_32o_e5_v8\01?0l" = linkonce_odr hidden unnamed_addr constant { i64, i64, i8*, i8*, i8*, {{.*}} } { i64 0, i64 40, i8* bitcast ({{.*}} to i8*), i8* bitcast ({{.*}} to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @[[STR]], i32 0, i32 0), {{.*}} }, align 8

typedef void (^BlockTy)(void);

void test(id a) {
  BlockTy b = ^{ (void)a; };
}
