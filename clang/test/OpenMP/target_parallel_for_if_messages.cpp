// RUN: %clang_cc1 -verify -fopenmp %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  int i;
  #pragma omp target parallel for if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel for' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target parallel for' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (S) // expected-error {{'S' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(argc)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(target : argc)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(parallel : argc)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(target : argc) if(parallel : argc)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(parallel : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp target parallel for'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(target : argc) if (target :argc) // expected-error {{directive '#pragma omp target parallel for' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(parallel : argc) if (parallel :argc) // expected-error {{directive '#pragma omp target parallel for' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(target : argc) if (argc) // expected-error {{expected  'parallel' directive name modifier}} expected-note {{previous clause with directive name modifier specified here}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(target : argc) if(parallel : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}} expected-note {{previous clause with directive name modifier specified here}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

int main(int argc, char **argv) {
  int i;
  #pragma omp target parallel for if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel for' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target parallel for' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(parallel : argc)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(target : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp target parallel for'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(target : argc) if (target :argc) // expected-error {{directive '#pragma omp target parallel for' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(parallel : argc) if (parallel :argc) // expected-error {{directive '#pragma omp target parallel for' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(target : argc) if (argc) // expected-error {{expected  'parallel' directive name modifier}} expected-note {{previous clause with directive name modifier specified here}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target parallel for if(target : argc) if(parallel : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}} expected-note {{previous clause with directive name modifier specified here}}
  for (i = 0; i < argc; ++i) foo();

  return tmain(argc, argv);
}
