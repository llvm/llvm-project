// RUN: %clang_cc1 -fsyntax-only -verify -Wreturn-type -Wuninitialized -Wno-switch-bool %s

enum E { A, B };

int f(E e) {
  switch (e) {
    case A: return 1;
    case B: return 2;
    default: ;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

int g(E e) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
  switch (e) {
    case A: x = 1; break;
    case B: x = 2; break;
    default: break; // expected-warning {{variable 'x' is used uninitialized whenever switch default is taken}}
  }
  return x; // expected-note {{uninitialized use occurs here}}
}

int h(E e) {
  switch (e) {
    case A: return 1;
    case B: return 2;
  }
} // no warning expected

int i(E e) {
  int x;
  switch (e) {
    case A: x = 1; break;
    case B: x = 2; break;
  }
  return x; // no warning expected
}

enum class BoolEnum : bool { False = false, True = true };

int test_bool_f(BoolEnum b) {
  switch (b) {
    case BoolEnum::True: return 1;
    case BoolEnum::False: return 2;
    default: ;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

int test_bool_g(BoolEnum b) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
  switch (b) {
    case BoolEnum::True: x = 1; break;
    case BoolEnum::False: x = 2; break;
    default: break; // expected-warning {{variable 'x' is used uninitialized whenever switch default is taken}}
  }
  return x; // expected-note {{uninitialized use occurs here}}
}

int test_bool_h(BoolEnum b) {
  switch (b) {
    case BoolEnum::True: return 1;
    case BoolEnum::False: return 2;
  }
} // no warning expected

int test_bool_i(BoolEnum b) {
  int x;
  switch (b) {
    case BoolEnum::True: x = 1; break;
    case BoolEnum::False: x = 2; break;
  }
  return x; // no warning expected
}
