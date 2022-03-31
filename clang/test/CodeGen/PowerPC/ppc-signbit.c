// RUN: %clang_cc1 %s -emit-llvm -o - -triple powerpc64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple powerpc64le-linux-gnu | FileCheck %s

int test(long double x) { return __builtin_signbitl(x); }

// CHECK-LABEL: define{{.*}} signext i32 @test(ppc_fp128 noundef %x)
// CHECK: bitcast ppc_fp128 %{{.*}} to i128
// CHECK: trunc i128 %{{.*}} to i64
// CHECK: icmp slt i64 %{{.*}}, 0
// CHECK: zext i1 %{{.*}} to i32

