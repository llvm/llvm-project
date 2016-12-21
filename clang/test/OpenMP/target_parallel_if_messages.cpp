// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  #pragma omp target parallel if // expected-error {{expected '(' after 'if'}}
  foo();
  #pragma omp target parallel if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if () // expected-error {{expected expression}}
  foo();
  #pragma omp target parallel if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel if (argc > 0 ? argv[1] : argv[2])
  foo();
  #pragma omp target parallel if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target parallel' cannot contain more than one 'if' clause}}
  foo();
  #pragma omp target parallel if (S) // expected-error {{'S' does not refer to a value}}
  foo();
  #pragma omp target parallel if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if(argc)
  foo();
  #pragma omp target parallel if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if(target : argc)
  foo();
  #pragma omp target parallel if(parallel : argc)
  foo();
  #pragma omp target parallel if(target : argc) if(parallel : argc)
  foo();
  #pragma omp target parallel if(parallel : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp target parallel'}}
  foo();
  #pragma omp target parallel if(target : argc) if (target :argc) // expected-error {{directive '#pragma omp target parallel' cannot contain more than one 'if' clause with 'target' name modifier}}
  foo();
  #pragma omp target parallel if(parallel : argc) if (parallel :argc) // expected-error {{directive '#pragma omp target parallel' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  foo();
  #pragma omp target parallel if(target : argc) if (argc) // expected-error {{expected  'parallel' directive name modifier}} expected-note {{previous clause with directive name modifier specified here}}
  foo();
  #pragma omp target parallel if(target : argc) if(parallel : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}} expected-note {{previous clause with directive name modifier specified here}}
  foo();

  return 0;
}

int main(int argc, char **argv) {
  #pragma omp target parallel if // expected-error {{expected '(' after 'if'}}
  foo();
  #pragma omp target parallel if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if () // expected-error {{expected expression}}
  foo();
  #pragma omp target parallel if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel if (argc > 0 ? argv[1] : argv[2])
  foo();
  #pragma omp target parallel if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target parallel' cannot contain more than one 'if' clause}}
  foo();
  #pragma omp target parallel if (S1) // expected-error {{'S1' does not refer to a value}}
  foo();
  #pragma omp target parallel if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel if(parallel : argc)
  foo();
  #pragma omp target parallel if(target : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp target parallel'}}
  foo();
  #pragma omp target parallel if(target : argc) if (target :argc) // expected-error {{directive '#pragma omp target parallel' cannot contain more than one 'if' clause with 'target' name modifier}}
  foo();
  #pragma omp target parallel if(parallel : argc) if (parallel :argc) // expected-error {{directive '#pragma omp target parallel' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  foo();
  #pragma omp target parallel if(target : argc) if (argc) // expected-error {{expected  'parallel' directive name modifier}} expected-note {{previous clause with directive name modifier specified here}}
  foo();
  #pragma omp target parallel if(target : argc) if(parallel : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}} expected-note {{previous clause with directive name modifier specified here}}
  foo();

  return tmain(argc, argv);
}
