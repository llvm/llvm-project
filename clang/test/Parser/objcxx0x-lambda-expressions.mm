// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused-value -Wno-c++1y-extensions -std=c++11 %s

class C {
  id get(int);

  void f() {
    int foo, bar, baz;

    // fail to parse as a lambda introducer, so we get objc message parsing errors instead
    [foo,+] {}; // expected-error {{expected expression}}

    []; // expected-error {{expected body of lambda expression}}
    [=,foo+] {}; // expected-error {{expected ',' or ']' in lambda capture list}}
    [&this] {}; // expected-error {{cannot take the address of an rvalue of type 'C *'}}
    [] {}; 
    [=] (int i) {}; 
    [&] (int) mutable -> void {}; 
    [foo,bar] () { return 3; }; 
    [=,&foo] () {}; 
    [this] () {}; 

    [foo(bar)] () {};
    [foo = bar] () {};
    [foo{bar}] () {};
    [foo = {bar}] () {}; // expected-error {{<initializer_list>}}

    [foo(bar) baz] () {}; // expected-error {{called object type 'int' is not a function}}
    [foo(bar), baz] () {}; // ok

    [foo = bar baz]; // expected-warning {{receiver type 'int'}} expected-warning {{instance method '-baz'}}

    [get(bar) baz]; // expected-warning {{instance method '-baz'}}
    [get(bar), baz]; // expected-error {{expected body of lambda}}

    [foo = bar ++ baz]; // expected-warning {{receiver type 'int'}} expected-warning {{instance method '-baz'}}
    [foo = bar + baz]; // expected-error {{expected body of lambda}}
    [foo = { bar, baz }]; // expected-error {{<initializer_list>}} expected-error {{expected body of lambda}}
    [foo = { bar } baz ]; // expected-warning {{receiver type 'int'}} expected-warning {{instance method '-baz'}}
    [foo = { bar }, baz ]; // expected-error {{<initializer_list>}} expected-error {{expected body of lambda}}
  }

};

struct Func {
  template <typename F>
  Func(F&&);
};

int getInt();

void test() {
  [val = getInt()]() { };
  Func{
    [val = getInt()]() { }
  };
}
