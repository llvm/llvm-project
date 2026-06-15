// RUN: %clang_cc1 -std=c++26 -freflection -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm -o - %s | FileCheck %s

using info = decltype(^^int);

template <auto A>
void foo () {}

int main() {
  foo <info {}> ();
  // CHECK: @_Z3fooITnDaLDmnuEEvv
  foo <^^void> ();
  // CHECK: @_Z3fooITnDaLDmtyvEEvv
  foo <^^bool> ();
  // CHECK: @_Z3fooITnDaLDmtybEEvv
  foo <^^char> ();
  // CHECK: @_Z3fooITnDaLDmtycEEvv
  foo <^^signed char> ();
  // CHECK: @_Z3fooITnDaLDmtyaEEvv
  foo <^^unsigned char> ();
  // CHECK: @_Z3fooITnDaLDmtyhEEvv
  foo <^^short> ();
  // CHECK: @_Z3fooITnDaLDmtysEEvv
  foo <^^unsigned short> ();
  // CHECK: @_Z3fooITnDaLDmtytEEvv
  foo <^^int> ();
  // CHECK: @_Z3fooITnDaLDmtyiEEvv
  foo <^^unsigned int> ();
  // CHECK: @_Z3fooITnDaLDmtyjEEvv
  foo <^^long> ();
  // CHECK: @_Z3fooITnDaLDmtylEEvv
  foo <^^unsigned long> ();
  // CHECK: @_Z3fooITnDaLDmtymEEvv
  foo <^^long long> ();
  // CHECK: @_Z3fooITnDaLDmtyxEEvv
  foo <^^unsigned long long> ();
  // CHECK: @_Z3fooITnDaLDmtyyEEvv
  foo <^^float> ();
  // CHECK: @_Z3fooITnDaLDmtyfEEvv
  foo <^^double> ();
  // CHECK: @_Z3fooITnDaLDmtydEEvv
  foo <^^long double> ();
  // CHECK: @_Z3fooITnDaLDmtyeEEvv
  return 0;
}
