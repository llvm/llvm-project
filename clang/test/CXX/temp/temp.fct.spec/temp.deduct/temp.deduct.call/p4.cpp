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
  struct B { char* x; };
  void bar() {
    fun(&B::x);
    // expected-error@-1 {{no matching function for call to 'fun'}}
    // expected-note@-9  {{candidate template ignored: could not match 'const int' against 'char'}}
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
  // We consider this to be a defect, and allow deduction to succeed in this
  // case. FIXME: Check this should be accepted once the DR is resolved.
  template<typename R> void baz(R(&)());
  void g() {
    baz(f);
  }

  // But there is one for member pointers.
  template<typename R, typename C, typename ...A> void quux(R (C::*)(A...));
  struct Q { void f(int, char) noexcept { quux(&Q::f); } };

  void g1() noexcept;
  void g2();
  template <class T> int h(T *, T *); // expected-note {{deduced conflicting types for parameter 'T' ('void () noexcept' vs. 'void ()')}}
  int x = h(g1, g2); // expected-error {{no matching function}}

  // We consider it a defect that deduction does not support the following.
  // FIXME: Check that the defect is resolved as we expect.
  template<bool B> int i(void () noexcept(B));
  int i1 = i(g1);
  int i2 = i(g2);
}
#endif
