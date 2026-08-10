// RUN: %clang_cc1 -fmodules -std=c++14 %s -verify
// expected-no-diagnostics

#pragma clang module build A
module A {}
#pragma clang module contents
#pragma clang module begin A
template<typename T> struct A {
  friend A operator+(const A&, const A&) { return {}; }
  template<typename T2> friend void func_1(const A&, const T2 &) {}
};
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
module B {}
#pragma clang module contents
#pragma clang module begin B
#pragma clang module import A
inline void f() { A<int> a; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build C
module C {}
#pragma clang module contents
#pragma clang module begin C
#pragma clang module import A
inline void g() { A<int> a; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import A
#pragma clang module import B
#pragma clang module import C

void h() {
  A<int> a;
  a + a;
  func_1(a, 0);
}
