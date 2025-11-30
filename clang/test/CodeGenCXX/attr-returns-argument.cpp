// RUN: %clang_cc1 -triple=x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

[[clang::returns_argument(1)]] int* test3(int* i) {
  // CHECK: @_Z5test3Pi(ptr noundef returned
  return i;
}

[[clang::returns_argument(1)]] int& test4(int& i) {
  // CHECK: @_Z5test4Ri(ptr noundef nonnull returned
  return i;
}

[[clang::returns_argument(1)]] int* test5(int* const i) {
  // CHECK: _Z5test5Pi(ptr noundef returned
  return i;
}

[[clang::returns_argument(2)]] int* test6(int*, int* i) {
  // CHECK: @_Z5test6PiS_(ptr noundef %{{.*}}, ptr noundef returned
  return i;
}

[[clang::returns_argument(1)]] int& test7(int* i) {
  //CHECK: @_Z5test7Pi(ptr noundef returned
  return *i;
}

struct S {
  [[clang::returns_argument(1)]] S& func1();
  [[clang::returns_argument(1)]] static S& func4(S*);
};

S& S::func1() {
  // CHECK: @_ZN1S5func1Ev(ptr noundef nonnull returned
  return *this;
}

S& S::func4(S* i) {
  // CHECK: @_ZN1S5func4EPS_(ptr noundef returned
  return *i;
}
