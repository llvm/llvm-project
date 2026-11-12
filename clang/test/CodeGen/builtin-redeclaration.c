// RUN: %clang_cc1 -emit-llvm-only %s

// PR45410
// Ensure we mark local extern redeclarations with a different type as non-builtin.
void non_builtin() {
  extern float exp();
  exp(); // Will crash due to wrong number of arguments if this calls the builtin.
}

// PR45410
// We mark exp() builtin as const with -fno-math-errno (default).
// We mustn't do that for extern redeclarations of builtins where the type differs.
float attribute() {
  extern float exp();
  return exp(1);
}
