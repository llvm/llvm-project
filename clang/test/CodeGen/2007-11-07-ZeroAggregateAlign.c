// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
struct A { short s; short t; int i; };
// CHECK: %a = alloca %struct.A, align 4
// // CHECK: call void @llvm.memset.p0i8.i{{[0-9]*}}(i8* align 4 %{{[0-9]*}}, {{.*}}, i1 false)
void q() { struct A a = {0}; }
