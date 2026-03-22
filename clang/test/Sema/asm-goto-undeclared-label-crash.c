// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test that we don't crash when an asm goto references an undeclared label
// and there's a variable with __attribute__((cleanup)) in scope.
// See: https://github.com/llvm/llvm-project/issues/175314

void a(int *b) {
  int __attribute__((cleanup(a))) c = 0; // expected-note {{jump exits scope of variable with __attribute__((cleanup))}}
  __asm__ goto("" : : : : d); // expected-error {{use of undeclared label 'd'}} \
                              // expected-error {{cannot jump from this asm goto statement to one of its possible targets}} \
                              // expected-note {{possible target of asm goto statement}}
}
