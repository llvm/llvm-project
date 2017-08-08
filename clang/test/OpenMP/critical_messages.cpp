// RUN: %clang_cc1 -verify -fopenmp %s

int foo();

template<typename T, int N>
int tmain(int argc, char **argv) { // expected-note {{declared here}}
  #pragma omp critical
  ;
  #pragma omp critical untied // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp critical'}}
  #pragma omp critical unknown // expected-warning {{extra tokens at the end of '#pragma omp critical' are ignored}}
  #pragma omp critical ( // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical ( + // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical (name2 // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical (name1)
  foo();
  {
    #pragma omp critical
  } // expected-error {{expected statement}}
  #pragma omp critical (name2) // expected-note {{previous 'critical' region starts here}}
  #pragma omp critical
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      foo();
      #pragma omp critical(name2) // expected-error {{cannot nest 'critical' regions having the same name 'name2'}}
      foo();
    }
  }
  #pragma omp critical (name2)
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
  #pragma omp critical (name2)
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

  #pragma omp critical (name2) hint // expected-error {{expected '(' after 'hint'}}
  foo();
  #pragma omp critical (name2) hint( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp critical (name2) hint(+ // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp critical (name2) hint(argc) // expected-error {{expression is not an integral constant expression}} expected-note {{read of non-const variable 'argc' is not allowed in a constant expression}}
  foo();
  #pragma omp critical (name) hint(N) // expected-error {{argument to 'hint' clause must be a strictly positive integer value}} expected-error {{constructs with the same name must have a 'hint' clause with the same value}} expected-note {{'hint' clause with value '4'}}
  foo();
  #pragma omp critical hint(N) // expected-error {{the name of the construct must be specified in presence of 'hint' clause}}
  foo();
  return 0;
}

int main(int argc, char **argv) { // expected-note {{declared here}}
  #pragma omp critical
  ;
  #pragma omp critical untied // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp critical'}}
  #pragma omp critical unknown // expected-warning {{extra tokens at the end of '#pragma omp critical' are ignored}}
  #pragma omp critical ( // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical ( + // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical (name2 // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical (name1)
  foo();
  {
    #pragma omp critical
  } // expected-error {{expected statement}}
  #pragma omp critical (name2) // expected-note {{previous 'critical' region starts here}}
  #pragma omp critical
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      foo();
      #pragma omp critical(name2) // expected-error {{cannot nest 'critical' regions having the same name 'name2'}}
      foo();
    }
  }
  #pragma omp critical (name2)
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
  #pragma omp critical (name2)
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

  #pragma omp critical (name2) hint // expected-error {{expected '(' after 'hint'}}
  foo();
  #pragma omp critical (name2) hint( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp critical (name2) hint(+ // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp critical (name2) hint(argc) // expected-error {{expression is not an integral constant expression}} expected-note {{read of non-const variable 'argc' is not allowed in a constant expression}}
  foo();
  #pragma omp critical (name) hint(23) // expected-note {{previous 'hint' clause with value '23'}}
  foo();
  #pragma omp critical hint(-5) // expected-error {{argument to 'hint' clause must be a strictly positive integer value}}
  foo();
  #pragma omp critical hint(1) // expected-error {{the name of the construct must be specified in presence of 'hint' clause}}
  foo();
  return tmain<int, 4>(argc, argv) + tmain<float, -5>(argc, argv); // expected-note {{in instantiation of function template specialization 'tmain<float, -5>' requested here}} expected-note {{in instantiation of function template specialization 'tmain<int, 4>' requested here}}
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
