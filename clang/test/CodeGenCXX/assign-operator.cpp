// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -o - -std=c++11 |FileCheck %s

class x {
public: int operator=(int);
};
void a() {
  x a;
  a = 1u;
}

void f(int i, int j) {
  // CHECK: load i32
  // CHECK: load i32
  // CHECK: add nsw i32
  // CHECK: store i32
  // CHECK: store i32 17, ptr
  // CHECK: ret
  (i += j) = 17;
}

// Taken from g++.old-deja/g++.jason/net.C
namespace test1 {
  template <class T> void fn (T t) { }
  template <class T> struct A {
    void (*p)(T);
    A() { p = fn; }
  };

  A<int> a;
}

// Ensure that we use memcpy when we would have selected a trivial assignment
// operator, even for a non-trivially-copyable type.
struct A {
  A &operator=(const A&);
};
struct B {
  B(const B&);
  B &operator=(const B&) = default;
  int n;
};
struct C {
  A a;
  B b[16];
};
void b(C &a, C &b) {
  // CHECK: define {{.*}} @_ZN1CaSERKS_(
  // CHECK: call {{.*}} @_ZN1AaSERKS_(
  // CHECK-NOT: call {{.*}} @_ZN1BaSERKS_(
  // CHECK: call {{.*}} @{{.*}}memcpy
  // CHECK-NOT: call {{.*}} @_ZN1BaSERKS_(
  // CHECK: }
  a = b;
}
