// RUN: %clang_cc1 %std_cxx98-14 -fsyntax-only -verify=expected,precxx17 -Wno-constant-conversion %s
// RUN: %clang_cc1 %std_cxx98-14 -fsyntax-only -verify=expected,precxx17 -Wno-constant-conversion -Wno-deprecated -Wdeprecated-increment-bool %s
// RUN: %clang_cc1 %std_cxx17- -fsyntax-only -verify=expected,cxx17 -Wno-constant-conversion -Wno-deprecated -Wdeprecated-increment-bool %s

// RUN: %clang_cc1 %std_cxx98-14 -fsyntax-only -verify=expected,precxx17 -Wno-constant-conversion %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %std_cxx98-14 -fsyntax-only -verify=expected,precxx17 -Wno-constant-conversion -Wno-deprecated -Wdeprecated-increment-bool %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %std_cxx17- -fsyntax-only -verify=expected,cxx17 -Wno-constant-conversion -Wno-deprecated -Wdeprecated-increment-bool %s -fexperimental-new-constant-interpreter


// Bool literals can be enum values.
enum {
  ReadWrite = false,
  ReadOnly = true
};

// bool cannot be decremented, and gives a warning on increment
void test(bool b)
{
  ++b; // precxx17-warning {{incrementing expression of type bool is deprecated}} \
          cxx17-error {{ISO C++17 does not allow incrementing expression of type bool}}
  b++; // precxx17-warning {{incrementing expression of type bool is deprecated}} \
          cxx17-error {{ISO C++17 does not allow incrementing expression of type bool}}
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
