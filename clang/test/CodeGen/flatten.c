// RUN: %clang_cc1 -triple=x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s

void f(void) {}

__attribute__((flatten))
// CHECK: define{{.*}} void @g() [[FLATTEN_ATTR:#[0-9]+]]
void g(void) {
  // CHECK-NOT: call {{.*}} @f
  f();
}

// CHECK: attributes [[FLATTEN_ATTR]] = {{{.*}}flatten{{.*}}}
