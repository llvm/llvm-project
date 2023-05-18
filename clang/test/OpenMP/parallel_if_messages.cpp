// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

void xxx(int argc) {
  int cond; // expected-note {{initialize the variable 'cond' to silence this warning}}
#pragma omp parallel if(cond) // expected-warning {{variable 'cond' is uninitialized when used here}}
  for (int i = 0; i < 10; ++i)
    ;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  T z;
  #pragma omp parallel if // expected-error {{expected '(' after 'if'}}
  #pragma omp parallel if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if () // expected-error {{expected expression}}
  #pragma omp parallel if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  #pragma omp parallel if (argc > 0 ? argv[1] : argv[2] + z)
  #pragma omp parallel if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp parallel' cannot contain more than one 'if' clause}}
  #pragma omp parallel if (S) // expected-error {{'S' does not refer to a value}}
  #pragma omp parallel if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if(argc)
  #pragma omp parallel if(parallel // expected-error {{use of undeclared identifier 'parallel'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if(parallel : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if(parallel : argc)
  #pragma omp parallel if(parallel : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp parallel'}}
  #pragma omp parallel if(parallel : argc) if (parallel:argc) // expected-error {{directive '#pragma omp parallel' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  #pragma omp parallel if(parallel : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}}
  foo();

  return 0;
}

int main(int argc, char **argv) {
int z;
  #pragma omp parallel if // expected-error {{expected '(' after 'if'}}
  #pragma omp parallel if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if () // expected-error {{expected expression}}
  #pragma omp parallel if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  #pragma omp parallel if (argc > 0 ? argv[1] : argv[2])
  #pragma omp parallel if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp parallel' cannot contain more than one 'if' clause}}
  #pragma omp parallel if (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp parallel if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if(parallel // expected-error {{use of undeclared identifier 'parallel'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if(parallel : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel if(parallel : argc>> z)
  #pragma omp parallel if(parallel : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp parallel'}}
  #pragma omp parallel if(parallel : argc) if (parallel:argc)  // expected-error {{directive '#pragma omp parallel' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  #pragma omp parallel if(parallel : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}}
  foo();

  return tmain(argc, argv);
}
