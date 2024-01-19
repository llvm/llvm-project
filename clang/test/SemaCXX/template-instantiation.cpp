// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ast-dump %s | FileCheck %s
// expected-no-diagnostics

namespace GH76521 {

#define MYATTR __attribute__((preserve_most))

template <typename T>
void foo() {
  // CHECK: FunctionDecl {{.*}} foo 'void ()'
  auto l = []() __attribute__((preserve_most)) {};
  // CHECK: CXXMethodDecl {{.*}} operator() 'auto () __attribute__((preserve_most)) const' inline
  auto l2 = [](T t) __attribute__((preserve_most)) -> T { return t; };
  // CHECK: CXXMethodDecl {{.*}} operator() 'auto (int) const -> int __attribute__((preserve_most))':'auto (int) __attribute__((preserve_most)) const -> int' implicit_instantiation inline instantiated_fro
}

template <typename T>
void bar() {
  // CHECK: FunctionDecl {{.*}} bar 'void ()'
  auto l = []() MYATTR {};
  // CHECK: CXXMethodDecl {{.*}} operator() 'auto () __attribute__((preserve_most)) const' inline
}

int main() {
  foo<int>();
  bar<int>();
}

}
