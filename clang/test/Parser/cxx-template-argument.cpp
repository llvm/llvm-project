// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> struct A {};

// Check for template argument lists followed by junk
// FIXME: The diagnostics here aren't great...
A<int+> int x; // expected-error {{expected '>'}} expected-error {{expected unqualified-id}}
A<int x; // expected-error {{expected '>'}}

// PR8912
template <bool> struct S {};
S<bool(2 > 1)> s;

// Test behavior when a template-id is ended by a token which starts with '>'.
namespace greatergreater {
  template<typename T> struct S { S(); S(T); };
  void f(S<int>=0); // expected-error {{a space is required between a right angle bracket and an equals sign (use '> =')}}
  void f(S<S<int>>=S<int>()); // expected-error {{use '> >'}} expected-error {{use '> ='}}
  template<typename T> void t();
  void g() {
    void (*p)() = &t<int>;
    (void)(&t<int>==p); // expected-error {{use '> ='}}
    (void)(&t<int>>=p); // expected-error {{use '> >'}}
    (void)(&t<S<int>>>=p); // expected-error {{use '> >'}}
    (void)(&t<S<int>>==p); // expected-error {{use '> >'}} expected-error {{use '> ='}}
  }
}

namespace PR5925 {
  template <typename x>
  class foo { // expected-note {{here}}
  };
  void bar(foo *X) { // expected-error {{requires template arguments}}
  }
}

namespace PR13210 {
  template <class T>
  class C {}; // expected-note {{here}}

  void f() {
    new C(); // expected-error {{requires template arguments}}
  }
}
