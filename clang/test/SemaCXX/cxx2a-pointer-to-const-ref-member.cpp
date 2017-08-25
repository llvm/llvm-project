// RUN: %clang_cc1 -std=c++2a %s -verify

struct X {
  void ref() & {}
  void cref() const& {}
};

void test() {
  X{}.ref(); // expected-error{{cannot initialize object parameter of type 'X' with an expression of type 'X'}}
  X{}.cref(); // expected-no-error

  (X{}.*&X::ref)(); // expected-error-re{{pointer-to-member function type 'void (X::*)() {{.*}}&' can only be called on an lvalue}}
  (X{}.*&X::cref)(); // expected-no-error
}
