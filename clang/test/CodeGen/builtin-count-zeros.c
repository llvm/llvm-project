// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple arm-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-ARM

int a(int a) {return __builtin_ctz(a) + __builtin_clz(a);}
// CHECK: call i32 @llvm.cttz.i32({{.*}}, i1 true)
// CHECK: call i32 @llvm.ctlz.i32({{.*}}, i1 true)
// CHECK-ARM: call i32 @llvm.cttz.i32({{.*}}, i1 false)
// CHECK-ARM: call i32 @llvm.ctlz.i32({{.*}}, i1 false)
