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
