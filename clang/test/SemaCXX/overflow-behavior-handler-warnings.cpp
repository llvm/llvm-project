// RUN: %clang_cc1 -fsyntax-only -std=c++23 -fexperimental-overflow-behavior-types -Wunused-label -verify %s

// A non-dependent operand is diagnosed once, in the template definition.
template <typename T> int nondep(int a, T) {
  return __builtin_overflow_goto(a + 1, l); // expected-warning {{'__builtin_overflow_goto' has no effect}}
l:
  return 0;
}
template int nondep(int, int);
template int nondep(int, long);

// A dependent operand is never diagnosed, even in instantiations where it
// performs no checked operation.
template <typename T> T dep(T a) {
  return __builtin_overflow_goto(a + 1, l);
l:
  return 0;
}
template int dep(int);
template int __ob_trap dep(int __ob_trap);

// Lambdas cannot reach the enclosing function's handler.
int lambda(int __ob_trap a) {
  auto f = [&] { return a * 2; }; // expected-warning {{overflow in this lambda traps; it cannot jump to the enclosing function's overflow handler label '__overflow'}}
  auto g = [&] {
    int r = a * 3;
    return r;
  __overflow:
    return -1;
  };
  return f() + g();
__overflow: // expected-note {{enclosing function's overflow handler label is here}}
  return -1;
}

// Nor the label of an enclosing builtin.
int lambda_in_builtin(int __ob_trap a) {
  return __builtin_overflow_goto([&] { return a + 1; }(), l); // expected-warning {{has no effect}} expected-warning {{overflow in this lambda traps; it cannot jump to the enclosing function's overflow handler label 'l'}}
l: // expected-note {{enclosing function's overflow handler label is here}}
  return 0;
}
