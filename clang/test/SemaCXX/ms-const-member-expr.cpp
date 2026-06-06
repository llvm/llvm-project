// RUN: %clang_cc1 %s -std=c++11 -fms-compatibility -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++11 -fms-compatibility -fsyntax-only -verify -fexperimental-new-constant-interpreter

struct S {
  enum { E = 1 };
  static const int sdm = 1;
};

void f(S *s) {
  char array[s->E] = { 0 };
}

extern S *s;
constexpr int e1 = s->E;

S *side_effect();  // expected-note{{declared here}}
constexpr int e2 = // expected-error{{must be initialized by a constant expression}}
    side_effect()->E; // expected-note{{cannot be used in a constant expression}}

constexpr int e4 = s->sdm;
