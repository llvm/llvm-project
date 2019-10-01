// RUN: %clang_cc1 -std=c++11 %s -fsyntax-only -verify -fms-compatibility -fexceptions -fcxx-exceptions
// RUN: %clang_cc1 -std=c++17 %s -fsyntax-only -verify -fms-compatibility -fexceptions -fcxx-exceptions

// FIXME: Should -fms-compatibility soften these errors into warnings to match
// MSVC? In practice, MSVC never implemented dynamic exception specifiers, so
// there isn't much Windows code in the wild that uses them.
#if __cplusplus >= 201703L
// expected-error@+3 {{ISO C++17 does not allow dynamic exception specifications}}
// expected-note@+2 {{use 'noexcept(false)' instead}}
#endif
void f() throw(...) { }

namespace PR28080 {
struct S;           // expected-note {{forward declaration}}
#if __cplusplus >= 201703L
// expected-error@+3 {{ISO C++17 does not allow dynamic exception specifications}}
// expected-note@+2 {{use 'noexcept(false)' instead}}
#endif
void fn() throw(S); // expected-warning {{incomplete type}} expected-note{{previous declaration}}
void fn() throw();  // expected-warning {{does not match previous declaration}}
}

template <typename T> struct FooPtr {
  template <typename U> FooPtr(U *p) : m_pT(nullptr) {}

  template <>
      // FIXME: It would be better if this note pointed at the primary template
      // above.
      // expected-note@+1 {{previous declaration is here}}
  FooPtr(T *pInterface) throw() // expected-warning {{exception specification in declaration does not match previous declaration}}
      : m_pT(pInterface) {}

  T *m_pT;
};
struct Bar {};
template struct FooPtr<Bar>; // expected-note {{requested here}}
