// RUN: %clang_cc1 -fsyntax-only -verify %s

// http://llvm.org/PR7905
namespace PR7905 {
struct S; // expected-note {{forward declaration}}
void foo1() {
  (void)(S[]) {{3}}; // expected-error {{array has incomplete element type}}
}

template <typename T> struct M { T m; };
void foo2() {
  (void)(M<short> []) {{3}};
}
}
