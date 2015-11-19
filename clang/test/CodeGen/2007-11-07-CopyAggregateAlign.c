// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
struct A { char s, t, u, v; short a; };
// CHECK: %a = alloca %struct.A, align 2
// CHECK: %b = alloca %struct.A, align 2
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i{{[0-9]*}}(i8* align 2 %{{[0-9]*}}, i8* align 2 %{{[0-9]*}}, {{.*}}, i1 false)

void q() { struct A a, b; a = b; }
