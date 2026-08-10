// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

int left();
int right();

void test() {
  const int a = 0;
  const int b = 0;
  __builtin_choose_expr(false, left() ? a : a, (right(), b));
}

// CHECK-LABEL: define{{.*}} void @_Z4testv()
// CHECK-NOT: call{{.*}} @_Z4leftv()
// CHECK: call{{.*}} @_Z5rightv()
// CHECK: ret void
