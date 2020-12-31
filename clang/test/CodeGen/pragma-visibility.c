// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

#pragma GCC visibility push(hidden)
int x = 2;
// CHECK: @x = hidden global

extern int y;
#pragma GCC visibility pop
int y = 4;
// CHECK: @y = hidden global

#pragma GCC visibility push(hidden)
extern __attribute((visibility("default"))) int z;
int z = 0;
// CHECK: @z ={{.*}} global
#pragma GCC visibility pop

#pragma GCC visibility push(hidden)
void f() {}
// CHECK-LABEL: define hidden void @f

__attribute((visibility("default"))) void g();
void g() {}
// CHECK-LABEL: define{{.*}} void @g
