// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify %s

inline namespace N { // expected-warning{{inline namespaces are a C++11 feature}}
struct X {
  template<typename ...Args> // expected-warning{{variadic templates are a C++11 extension}}
  void f(Args &&...) &; // expected-warning{{rvalue references are a C++11 extension}} \
  // expected-warning{{reference qualifiers on functions are a C++11 extension}}
};
}

struct B {
  virtual void f();
  virtual void g();
};
struct D final : B { // expected-warning {{'final' keyword is a C++11 extension}}
  virtual void f() override; // expected-warning {{'override' keyword is a C++11 extension}}
  virtual void g() final; // expected-warning {{'final' keyword is a C++11 extension}}
};

void NewBracedInitList() {
  // A warning on this would be sufficient once we can handle it correctly.
  new int {}; // expected-error {{}}
}

struct Auto {
  static int n;
};
auto Auto::n = 0; // expected-warning {{'auto' type specifier is a C++11 extension}}
auto Auto::m = 0; // expected-error {{no member named 'm' in 'Auto'}}
                  // expected-warning@-1 {{'auto' type specifier is a C++11 extension}}

struct Conv { template<typename T> operator T(); };
bool pr21367_a = new int && false;
bool pr21367_b = &Conv::operator int && false;
