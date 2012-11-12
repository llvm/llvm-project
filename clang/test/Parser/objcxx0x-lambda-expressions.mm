// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused-value -std=c++11 %s

class C {

  void f() {
    int foo, bar;

    // fail to parse as a lambda introducer, so we get objc message parsing errors instead
    [foo,+] {}; // expected-error {{expected expression}}

    []; // expected-error {{expected body of lambda expression}}
    [=,foo+] {}; // expected-error {{expected ',' or ']' in lambda capture list}}
    [&this] {}; // expected-error {{address expression must be an lvalue}}
    [] {}; 
    [=] (int i) {}; 
    [&] (int) mutable -> void {}; 
    [foo,bar] () { return 3; }; 
    [=,&foo] () {}; 
    [this] () {}; 
  }

};

