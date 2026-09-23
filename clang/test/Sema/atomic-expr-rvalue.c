// RUN: %clang_cc1 -fsyntax-only %s -verify
// expected-no-diagnostics

typedef _Atomic char atomic_char;
atomic_char counter;

// Check correct implicit conversions of r-value atomic expressions.
// Bugfix: https://github.com/llvm/llvm-project/issues/106576
char load_plus_one_stmtexpr() {
  return ({counter;}) + 1;
}

char load_stmtexpr() {
  return ({counter;});
}

char load_cast_plus_one() {
  return (atomic_char)('x') + 1;
}

char load_cast() {
  return (atomic_char)('x');
}
