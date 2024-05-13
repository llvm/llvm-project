// RUN: %clang_cc1 %std_cxx98- -fcxx-exceptions -fexceptions -Wno-dynamic-exception-spec %s -triple=i686-unknown-linux -emit-llvm -o - | FileCheck %s
typedef int Array[10];

void foo() throw (Array) {
  throw 0;
  // CHECK: landingpad
  // CHECK-NEXT: filter {{.*}} @_ZTIPi
}

struct S {
  void foo() throw (S[10]) {
    throw 0;
  }
};

template <typename T>
struct S2 {
  void foo() throw (T) {
    throw 0;
  }
};

int main() {
  S s;
  s.foo();
  // CHECK: landingpad
  // CHECK-NEXT: filter {{.*}} @_ZTIP1S

  S2 <int[10]> s2;
  s2.foo();
  // CHECK: landingpad
  // CHECK-NEXT: filter {{.*}} @_ZTIPi
}
