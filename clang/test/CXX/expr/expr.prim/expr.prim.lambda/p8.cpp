// RUN: %clang_cc1 -std=c++11 %s -verify

class X0 {
  void explicit_capture() {
    int foo;

    (void)[foo, foo] () {}; // expected-error {{'foo' can appear only once}}
    (void)[this, this] () {}; // expected-error {{'this' can appear only once}}
    (void)[=, foo] () {}; // expected-error {{'&' must precede a capture when}}
    (void)[=, &foo] () {};
    (void)[=, this] () {}; // expected-error {{'this' cannot be explicitly captured}}
    (void)[&, foo] () {};
    (void)[&, &foo] () {}; // expected-error {{'&' cannot precede a capture when}} 
    (void)[&, this] () {};
  }
};

struct S2 { 
  void f(int i); 
  void g(int i);
};

void S2::f(int i) {
  (void)[&, i]{ };
  (void)[&, &i]{ }; // expected-error{{'&' cannot precede a capture when the capture default is '&'}}
  (void)[=, this]{ }; // expected-error{{'this' cannot be explicitly captured}}
  (void)[=]{ this->g(i); };
  (void)[i, i]{ }; // expected-error{{'i' can appear only once in a capture list}}
}
