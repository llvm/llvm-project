// RUN: %clang_cc1 -emit-llvm -std=c++11 -o - %s -triple x86_64-pc-linux-gnu | FileCheck %s

struct A {
  A &operator=(A&&);
};

struct B {
  A a;
  int i;
  bool b;
  char c;
  long l;
  float f;
};

void test1() {
  B b1, b2;
  b1 = static_cast<B&&>(b2);
}

// CHECK-LABEL: define {{.*}} @_ZN1BaSEOS_
// CHECK: call {{.*}} @_ZN1AaSEOS_
// CHECK-NOT: store
// CHECK: call {{.*}}memcpy{{.*}}, i64 24
// CHECK-NOT: store
// CHECK: ret
