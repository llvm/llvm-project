// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

void f(); // expected-note {{possible target for call}}
void f(int); // expected-note {{possible target for call}}

void g() {
  bool b = noexcept(f); // expected-error {{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}}
  bool b2 = noexcept(f(0));
}

struct S {
  void g(); // expected-note {{possible target for call}}
  void g(int); // expected-note {{possible target for call}}

  void h() {
    bool b = noexcept(this->g); // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
    bool b2 = noexcept(this->g(0));
  }
};
