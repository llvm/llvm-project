// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s
struct A { short s; short t; int i; };
// CHECK: %a = alloca %struct.A, align 4
// CHECK: call void @llvm.memset.p0i8.{{.*}} align 4 {{.*}}, i1 false)
void q(void) { struct A a = {0}; }
