// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

[[clang::default_constructed_after_move]] int a; // expected-error {{'clang::default_constructed_after_move' attribute only applies to classes}}

[[clang::default_constructed_after_move]] void f(); // expected-error {{only applies to}}

enum [[clang::default_constructed_after_move]] E { A, B }; // expected-error {{only applies to}}

void foo( [[clang::default_constructed_after_move]] int param); // expected-error {{only applies to}}

struct MyStruct {
  [[clang::default_constructed_after_move]] int member; // expected-error {{only applies to}}
};

class [[clang::default_constructed_after_move]] C {
public:
  C();
  C(C &&);
  C &operator=(C &&);
};

C [[clang::default_constructed_after_move]] c_var; // expected-error {{'clang::default_constructed_after_move' attribute cannot be applied to types}}

struct [[clang::default_constructed_after_move]] S {
  S();
  S(S &&);
  S &operator=(S &&);
};

union [[clang::default_constructed_after_move]] U {
  int a;
  float b;
  U(U &&);
  U &operator=(U &&);
};
