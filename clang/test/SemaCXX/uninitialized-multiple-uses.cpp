// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -verify %s

void use_val(int);
void use_const_ref(const int &);

// Test that the warning about self initialization is generated only once.
void test_self_init_1warning(bool a) {
  int v = v; // expected-warning {{variable 'v' is uninitialized when used within its own initialization}}
  if (a)
    use_val(v);
  else
    use_const_ref(v);
}

// Test that the diagnostic for using an uninitialized variable directly has a
// higher priority than using the same variable via a const reference.
void test_prioritize_use_over_const_ref(bool a) {
  int v; // expected-note {{initialize the variable 'v' to silence this warning}}
  if (a) // expected-warning {{variable 'v' is used uninitialized whenever 'if' condition is false}}
         // expected-note@-1 {{remove the 'if' if its condition is always true}}
    v = 2;
  else
    use_const_ref(v);
  use_val(v); // expected-note {{uninitialized use occurs here}}
}
