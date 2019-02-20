// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -ast-dump | FileCheck %s
template<typename T>
void f(T t) {
  T a[] = {t};
  for (auto x : a) {}
}

void g() {
  f(1);
}
// CHECK: VarDecl {{.*}} implicit used __range
// CHECK: VarDecl {{.*}} implicit used __range
// CHECK: VarDecl {{.*}} implicit used __begin
// CHECK: VarDecl {{.*}} implicit used __end
