// RUN: %clang_cc1 -verify -fopenmp %s

int foo();

int main() {
  #pragma omp critical
  ;
  #pragma omp critical untied // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp critical'}}
  #pragma omp critical unknown // expected-warning {{extra tokens at the end of '#pragma omp critical' are ignored}}
  #pragma omp critical ( // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical ( + // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical (name // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical (name1)
  foo();
  {
    #pragma omp critical
  } // expected-error {{expected statement}}
  #pragma omp critical (name) // expected-note {{previous 'critical' region starts here}}
  #pragma omp critical
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      foo();
      #pragma omp critical(name) // expected-error {{cannot nest 'critical' regions having the same name 'name'}}
      foo();
    }
  }
  #pragma omp critical (name)
  #pragma omp critical
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      #pragma omp critical
      foo();
    }
  }
  #pragma omp critical (name)
  #pragma omp critical
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      #pragma omp critical (nam)
      foo();
    }
  }

  return 0;
}

int foo() {
  L1:
    foo();
  #pragma omp critical
  {
    foo();
    goto L1; // expected-error {{use of undeclared label 'L1'}}
  }
  goto L2; // expected-error {{use of undeclared label 'L2'}}
  #pragma omp critical
  {
    L2:
    foo();
  }

  return 0;
}
