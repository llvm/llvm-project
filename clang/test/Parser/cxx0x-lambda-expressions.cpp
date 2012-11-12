// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -verify -std=c++11 %s

enum E { e };

class C {

  int f() {
    int foo, bar;

    []; // expected-error {{expected body of lambda expression}}
    [+] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}
    [foo+] {}; // expected-error {{expected ',' or ']' in lambda capture list}}
    [foo,&this] {}; // expected-error {{'this' cannot be captured by reference}}
    [&this] {}; // expected-error {{'this' cannot be captured by reference}}
    [&,] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}
    [=,] {}; // expected-error {{expected variable name or 'this' in lambda capture list}}
    [] {}; 
    [=] (int i) {}; 
    [&] (int) mutable -> void {}; 
    [foo,bar] () { return 3; }; 
    [=,&foo] () {}; 
    [&,foo] () {}; 
    [this] () {}; 
    [] () -> class C { return C(); };
    [] () -> enum E { return e; };

    [] -> int { return 0; }; // expected-error{{lambda requires '()' before return type}}
    [] mutable -> int { return 0; }; // expected-error{{lambda requires '()' before 'mutable'}}
    [](int) -> {}; // PR13652 expected-error {{expected a type}}
    return 1;
  }

  void designator_or_lambda() {
    typedef int T; 
    const int b = 0; 
    const int c = 1;
    int a1[1] = {[b] (T()) {}}; // expected-error{{no viable conversion from '<lambda}}
    int a2[1] = {[b] = 1 };
    int a3[1] = {[b,c] = 1 }; // expected-error{{expected body of lambda expression}}
    int a4[1] = {[&b] = 1 }; // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'const int *'}}
    int a5[3] = { []{return 0;}() };
    int a6[1] = {[this] = 1 }; // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'C *'}}
  }

  void delete_lambda(int *p) {
    delete [] p;
    delete [] (int*) { new int }; // ok, compound-literal, not lambda
    delete [] { return new int; } (); // expected-error{{expected expression}}
    delete [&] { return new int; } (); // ok, lambda
  }
};
