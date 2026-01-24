// RUN: %clang_cc1 -std=c11 -fsyntax-only -fexperimental-new-constant-interpreter %s

void foo(void) {
  // Complex comparison evaluated in a discarded context.
  (void)(0 && (1i == 1i));
}
