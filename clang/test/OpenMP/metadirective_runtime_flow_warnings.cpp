// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -std=c++11 \
// RUN:   -fopenmp-metadirective-runtime-selection -Wuninitialized \
// RUN:   -Wunreachable-code -fsyntax-only %s

[[noreturn]] void stop();
void use(int);

void plain_parallel() {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp parallel
  {
    use(x); // expected-warning {{variable 'x' is uninitialized when used here}}
  }
}

// FIXME: Known limitation, to be fixed in a follow-up: CFG-based analyses
// skip the variants, so this does not warn.
void runtime_metadirective(int flag) {
  int x;
#pragma omp metadirective when(user={condition(flag)}: parallel) otherwise(single)
  {
    use(x);
  }
}

void unreachable_parallel() {
#pragma omp parallel
  {
    stop();
    use(1); // expected-warning {{code will never be executed}}
  }
}

// FIXME: Known limitation, to be fixed in a follow-up: CFG-based analyses
// skip the variants, so this does not warn.
void unreachable_metadirective(int flag) {
#pragma omp metadirective when(user={condition(flag)}: parallel) otherwise(single)
  {
    stop();
    use(1);
  }
}
