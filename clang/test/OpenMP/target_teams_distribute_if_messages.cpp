// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  int i;
#pragma omp target teams distribute if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target teams distribute' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (S) // expected-error {{'S' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(argc)
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(target : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(target : argc)
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(teams: argc) // expected-error {{directive name modifier 'teams' is not allowed for '#pragma omp target teams distribute'}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(distribute: argc) // expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute'}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(argc) if(teams: argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{directive name modifier 'teams' is not allowed for '#pragma omp target teams distribute'}} expected-error {{no more 'if' clause is allowed}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(argc) if(distribute: argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute'}} expected-error {{no more 'if' clause is allowed}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(distribute: argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp target teams distribute'}} expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute'}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(target : argc) if (target :argc) // expected-error {{directive '#pragma omp target teams distribute' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(target : argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{no more 'if' clause is allowed}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(target : argc) if(distribute: argc) if (argc) // expected-note 2 {{previous clause with directive name modifier specified here}} expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute'}} expected-error {{expected one of  directive name modifiers}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

int main(int argc, char **argv) {
  int i;
#pragma omp target teams distribute if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target teams distribute' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(target : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(teams : argc) // expected-error {{directive name modifier 'teams' is not allowed for '#pragma omp target teams distribute'}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(distribute : argc) // expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute'}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(target : argc) if (distribute :argc) // expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute'}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(target : argc) if (target :argc) // expected-error {{directive '#pragma omp target teams distribute' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(target : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(argc) if(teams: argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{directive name modifier 'teams' is not allowed for '#pragma omp target teams distribute'}} expected-error {{no more 'if' clause is allowed}}
  for (i = 0; i < argc; ++i) foo();

#pragma omp target teams distribute if(argc) if(distribute: argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute'}} expected-error {{no more 'if' clause is allowed}}
  for (i = 0; i < argc; ++i) foo();

  return tmain(argc, argv);
}
