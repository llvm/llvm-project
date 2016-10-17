// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z %s

namespace PR8598 {
  template<class T> struct identity { typedef T type; };

  template<class T, class C>
  void f(T C::*, typename identity<T>::type*){}
  
  struct X { void f() {}; };
  
  void g() { (f)(&X::f, 0); }
}

namespace PR12132 {
  template<typename S> void fun(const int* const S::* member) {}
  struct A { int* x; };
  void foo() {
    fun(&A::x);
  }
}

#if __cplusplus > 201402L
namespace noexcept_conversion {
  template<typename R> void foo(R());
  template<typename R> void bar(R()) = delete;
  template<typename R> void bar(R() noexcept) {}
  void f() throw() {
    foo(&f);
    bar(&f);
  }
  // There is no corresponding rule for references.
  // FIXME: This seems like a defect.
  template<typename R> void baz(R(&)()); // expected-note {{does not match adjusted type}}
  void g() {
    baz(f); // expected-error {{no match}}
  }

  void g1() noexcept;
  void g2();
  template <class T> int h(T *, T *); // expected-note {{deduced conflicting types for parameter 'T' ('void () noexcept' vs. 'void ()')}}
  int x = h(g1, g2); // expected-error {{no matching function}}
}
#else
// expected-no-diagnostics
#endif
