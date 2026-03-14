// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98 -Wno-c++11-extensions

struct S {
  int *begin();
  int *end();
};

struct T {
};

struct Range {};
int begin(Range); // expected-note {{not viable}}
int end(Range);

namespace NS {
  struct ADL {};
  struct iter {
    int operator*();
    bool operator!=(iter);
    void operator++();
  };
  iter begin(ADL); // expected-note {{not viable}}
  iter end(ADL);

  struct NoADL {};
}
NS::iter begin(NS::NoADL); // expected-note {{not viable}}
NS::iter end(NS::NoADL);

void f() {
  int a[] = {1, 2, 3};
  for (auto b : S()) {} // ok
  for (auto b : T()) {} // expected-error {{invalid range expression of type 'T'}}
  for (auto b : a) {} // ok
  for (int b : NS::ADL()) {} // ok
  for (int b : NS::NoADL()) {} // expected-error {{invalid range expression of type 'NS::NoADL'}}
}

void PR11601() {
  void (*vv[])() = {PR11601, PR11601, PR11601};
  for (void (*i)() : vv) i();
}
