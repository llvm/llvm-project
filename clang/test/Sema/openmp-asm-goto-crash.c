// RUN: %clang_cc1 -fopenmp -fsyntax-only -verify -triple x86_64-unknown-linux-gnu %s

int test_asm_goto_undeclared_label() {
#pragma omp assume
  __asm__ goto("" : : : : undefined_label); // expected-error {{use of undeclared label 'undefined_label'}} \
                                             // expected-error {{cannot jump from this asm goto statement to one of its possible targets}} \
                                             // expected-note {{jump exits scope of OpenMP structured block}}
  int x = 1;
undefined_label:
  return x;
}
