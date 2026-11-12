// RUN: %clang_cc1 -fsyntax-only -verify %s

class ctor {
  ctor(); // expected-note{{implicitly declared private here}}
};

class dtor {
  ~dtor(); // expected-note 3 {{implicitly declared private here}}
};

void test() {
  new ctor[0]; // expected-error{{calling a private constructor of class 'ctor'}}
  new dtor[0]; // expected-error{{temporary of type 'dtor' has private destructor}}
  new dtor[3]; // expected-error{{temporary of type 'dtor' has private destructor}}
  new dtor[3][3]; // expected-error{{temporary of type 'dtor' has private destructor}}
}
