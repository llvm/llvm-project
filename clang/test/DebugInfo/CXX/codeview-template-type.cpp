// RUN: %clang -std=c++11 -g -target x86_64-windows-msvc -S -emit-llvm -o - %s | FileCheck %s

template <unsigned N>
void foo() {
}

void instantiate_foo() {
  foo<10>();
  // CHECK: foo<10>
  foo<true>();
  // CHECK: foo<1>
}
