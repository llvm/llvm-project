// Test without serialization:
// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -ast-dump | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++11 -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

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
