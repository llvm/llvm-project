// RUN: %clang_cc1 -std=c++14 -verify %s
// RUN: %clang_cc1 -std=c++1z -verify %s

#if __cplusplus > 201402L

template<typename T> void redecl1() noexcept(noexcept(T())) {} // expected-note {{previous}}
template<typename T> void redecl1() noexcept(noexcept(T())); // ok, same type
template<typename T> void redecl1() noexcept(noexcept(T())) {} // expected-error {{redefinition}}

template<bool A, bool B> void redecl2() noexcept(A); // expected-note {{previous}}
template<bool A, bool B> void redecl2() noexcept(B); // expected-error {{conflicting types}}

// These have the same canonical type.
// FIXME: It's not clear whether this is supposed to be valid.
template<typename A, typename B> void redecl3() throw(A);
template<typename A, typename B> void redecl3() throw(B);

typedef int I;
template<bool B> void redecl4(I) noexcept(B);
template<bool B> void redecl4(I) noexcept(B);

namespace DependentDefaultCtorExceptionSpec {
  template<typename> struct T { static const bool value = true; };

  template<class A> struct map {
    typedef A a;
    map() noexcept(T<a>::value) {}
  };

  template<class B> struct multimap {
    typedef B b;
    multimap() noexcept(T<b>::value) {}
  };

  // Don't crash here.
  struct A { multimap<int> Map; } a;

  static_assert(noexcept(A()));
}

#endif

namespace CompatWarning {
  struct X;

  // These cases don't change.
  void f0(void p() throw(int));
  auto f0() -> void (*)() noexcept(false);

  // These cases take an ABI break in C++17 because their parameter / return types change.
  void f1(void p() noexcept);
  void f2(void (*p)() noexcept(true));
  void f3(void (&p)() throw());
  void f4(void (X::*p)() throw());
  auto f5() -> void (*)() throw();
  auto f6() -> void (&)() throw();
  auto f7() -> void (X::*)() throw();
#if __cplusplus <= 201402L
  // expected-warning@-8 {{mangled name of 'f1' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f2' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f3' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f4' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f5' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f6' will change in C++17 due to non-throwing exception specification in function signature}}
  // expected-warning@-8 {{mangled name of 'f7' will change in C++17 due to non-throwing exception specification in function signature}}
#endif

  // An instantiation-dependent exception specification needs to be mangled in
  // all language modes, since it participates in SFINAE.
  template<typename T> void g(void() throw(T)); // expected-note {{substitution failure}}
  template<typename T> void g(...) = delete; // expected-note {{deleted}}
  void test_g() { g<void>(nullptr); } // expected-error {{deleted}}

  // An instantiation-dependent exception specification needs to be mangled in
  // all language modes, since it participates in SFINAE.
  template<typename T> void h(void() noexcept(T())); // expected-note {{substitution failure}}
  template<typename T> void h(...) = delete; // expected-note {{deleted}}
  void test_h() { h<void>(nullptr); } // expected-error {{deleted}}
}
