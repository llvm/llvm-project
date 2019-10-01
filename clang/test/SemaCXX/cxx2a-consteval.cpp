// RUN: %clang_cc1 -std=c++2a -fsyntax-only %s -verify

namespace basic_sema {

consteval int f1(int i) {
  return i;
}

consteval constexpr int f2(int i) { 
  //expected-error@-1 {{cannot combine}}
  return i;
}

constexpr auto l_eval = [](int i) consteval {

  return i;
};

constexpr consteval int f3(int i) {
  //expected-error@-1 {{cannot combine}}
  return i;
}

struct A {
  consteval int f1(int i) const {
    return i;
  }
  consteval A(int i);
  consteval A() = default;
  consteval ~A() = default;
};

consteval struct B {}; // expected-error {{struct cannot be marked consteval}}

consteval typedef B b; // expected-error {{typedef cannot be consteval}}

consteval int redecl() {return 0;} // expected-note {{previous declaration is here}}
constexpr int redecl() {return 0;} // expected-error {{constexpr declaration of 'redecl' follows consteval declaration}}

consteval int i = 0; // expected-error {{consteval can only be used in function declarations}}

consteval int; // expected-error {{consteval can only be used in function declarations}}

consteval int f1() {} // expected-error {{no return statement in consteval function}}

struct C {
  C() {}
  ~C() {}
};

struct D {
  C c;
  consteval D() = default; // expected-error {{cannot be consteval}}
  consteval ~D() = default; // expected-error {{cannot be consteval}}
};

struct E : C { // expected-note {{here}}
  consteval ~E() {} // expected-error {{cannot be declared consteval because base class 'basic_sema::C' does not have a constexpr destructor}}
};
}

consteval int main() { // expected-error {{'main' is not allowed to be declared consteval}}
  return 0;
}
