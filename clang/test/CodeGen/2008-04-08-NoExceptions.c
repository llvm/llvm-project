// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s %}

void f(void);
void g(void) {
  // CHECK: define{{.*}} void @g() [[NUW:#[0-9]+]]
  // CHECK-NOT: call void @f() nounwind
  f();
}

// CHECK-NOT: declare void @f() [[NUW]]

// CHECK: attributes [[NUW]] = { noinline nounwind{{.*}} }
