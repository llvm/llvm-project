// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused-value -std=c++20 %s
// expected-no-diagnostics

namespace GH64347 {

template<typename X, typename Y> struct A { X x; Y y;};
void test() {
   A(1, 2);
   new A(1, 2);
}

template<A a>
void f() { (void)a; }
void k() {
  // Test CTAD works for non-type template arguments.
  f<A(0, 0)>();
}

} // namespace GH64347
