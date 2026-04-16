// RUN: %clang_cc1 -std=c++26 -freflection -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm -o - %s | FileCheck %s

using info = decltype(^^int);

template <auto A>
void foo () {}

int main() {
  foo <info {}> ();
  foo <^^int> ();
  return 0;
}
// CHECK: define linkonce_odr void @_Z3fooITnDaLDmnuEEvv()
// CHECK: define linkonce_odr void @_Z3fooITnDaLDmtyiEEvv()
