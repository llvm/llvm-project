// RUN: %clang_cc1 -fsyntax-only -verify -Wno-constant-conversion %s

// Bool literals can be enum values.
enum {
  ReadWrite = false,
  ReadOnly = true
};

// bool cannot be decremented, and gives a warning on increment
void test(bool b)
{
  ++b; // expected-warning {{incrementing expression of type bool is deprecated}}
  b++; // expected-warning {{incrementing expression of type bool is deprecated}}
  --b; // expected-error {{cannot decrement expression of type bool}}
  b--; // expected-error {{cannot decrement expression of type bool}}

  bool *b1 = (int *)0; // expected-error{{cannot initialize}}
}

// static_assert_arg_is_bool(x) compiles only if x is a bool.
template <typename T>
void static_assert_arg_is_bool(T x) {
  bool* p = &x;
}

void test2() {
  int n = 2;
  static_assert_arg_is_bool(n && 4);  // expected-warning {{use of logical '&&' with constant operand}} \
                                      // expected-note {{use '&' for a bitwise operation}} \
                                      // expected-note {{remove constant to silence this warning}}
  static_assert_arg_is_bool(n || 5);  // expected-warning {{use of logical '||' with constant operand}} \
                                      // expected-note {{use '|' for a bitwise operation}}
}
