// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

// PR6141
template<typename T>
struct X {
  X();
  template<typename U> X(X<U>);
  X(const X<T>&);
};

void f(X<int>) { }

struct Y : X<int> { };
struct Z : X<float> { };

// CHECK: define i32 @main()
int main() {
  // CHECK: call void @_ZN1YC1Ev
  // CHECK: call void @_ZN1XIiEC1ERKS0_
  // CHECK: call void @_Z1f1XIiE
  f(Y());
  // CHECK: call void @_ZN1ZC1Ev
  // CHECK: call void @_ZN1XIfEC1ERKS0_
  // CHECK: call void @_ZN1XIiEC1IfEES_IT_E
  // CHECK: call void @_Z1f1XIiE
  f(Z());
}
