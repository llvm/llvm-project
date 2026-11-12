// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -emit-llvm -fobjc-runtime=gnustep-1.7 -fblocks -o - %s | FileCheck --check-prefix=CHECK-COMDAT %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -fobjc-runtime=gcc -fblocks -o - %s | FileCheck --check-prefix=CHECK-COMDAT %s
// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm -fblocks -o - %s | FileCheck %s

// Test that descriptor symbol names don't include '@'.

// CHECK-COMDAT: $"__block_descriptor_40_8_32o_e5_v8\01?0l" = comdat any
// CHECK-COMDAT: @[[STR:.*]] = private unnamed_addr constant [6 x i8] c"v8@?0\00"
// CHECK-COMDAT: @"__block_descriptor_40_8_32o_e5_v8\01?0l" = linkonce_odr hidden unnamed_addr constant { i64, i64, ptr, ptr, ptr, {{.*}} } { i64 0, i64 40, ptr {{.*}}, ptr @[[STR]], {{.*}} }, comdat, align 8

// CHECK: @[[STR:.*]] = private unnamed_addr constant [6 x i8] c"v8@?0\00"
// CHECK: @"__block_descriptor_40_8_32o_e5_v8\01?0l" = linkonce_odr hidden unnamed_addr constant { i64, i64, ptr, ptr, ptr, {{.*}} } { i64 0, i64 40, ptr {{.*}}, ptr {{.*}}, ptr @[[STR]], {{.*}} }, align 8

typedef void (^BlockTy)(void);

void test(id a) {
  BlockTy b = ^{ (void)a; };
}
