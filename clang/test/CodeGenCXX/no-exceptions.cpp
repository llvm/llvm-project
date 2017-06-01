// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

void g();

// CHECK: define void @_Z1fv() [[NUW:#[0-9]+]]
void f() throw (int) { 

  // CHECK-NOT: invoke void @_Z1gv
  g();
  // CHECK: call void @_Z1gv()
  // CHECK: ret void
}

// CHECK: attributes [[NUW]] = { noinline nounwind{{.*}} }
