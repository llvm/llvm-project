// RUN: %clang_cc1 -fsyntax-only -Wself-assign -verify %s

void f() {
  int a = 42, b = 42;
  a = a; // expected-warning{{explicitly assigning}}
  b = b; // expected-warning{{explicitly assigning}}
  a = b;
  b = a = b;
  a = a = a; // expected-warning{{explicitly assigning}}
  a = b = b = a;
}

// Dummy type.
struct S {};

void false_positives() {
#define OP =
#define LHS a
#define RHS a
  int a = 42;
  // These shouldn't warn due to the use of the preprocessor.
  a OP a;
  LHS = a;
  a = RHS;
  LHS OP RHS;
#undef OP
#undef LHS
#undef RHS

  S s;
  s = s; // Not a builtin assignment operator, no warning.

  // Volatile stores aren't side-effect free.
  volatile int vol_a;
  vol_a = vol_a;
  volatile int &vol_a_ref = vol_a;
  vol_a_ref = vol_a_ref;
}

template <typename T> void g() {
  T a;
  a = a; // May or may not be a builtin assignment operator, no warning.
}
void instantiate() {
  g<int>();
  g<S>();
}
