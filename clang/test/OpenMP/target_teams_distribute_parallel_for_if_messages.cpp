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
#pragma omp target teams distribute parallel for if(cond) // expected-warning {{variable 'cond' is uninitialized when used here}}
  for (int i = 0; i < 10; ++i)
    ;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  T i, z;
#pragma omp target teams distribute parallel for if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute parallel for' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target teams distribute parallel for' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (S) // expected-error {{'S' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(argc -z)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel // expected-error {{use of undeclared identifier 'parallel'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : argc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp target teams distribute parallel for'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : argc) if (parallel:argc) // expected-error {{directive '#pragma omp target teams distribute parallel for' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(target : argc) if (target:argc) // expected-error {{directive '#pragma omp target teams distribute parallel for' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : argc) if (argc) // expected-error {{expected 'target' directive name modifier}} expected-note {{previous clause with directive name modifier specified here}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(target: argc) if (argc) // expected-error {{expected 'parallel' directive name modifier}} expected-note {{previous clause with directive name modifier specified here}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(distribute : argc) // expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute parallel for'}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

int main(int argc, char **argv) {
  int i, z;
#pragma omp target teams distribute parallel for if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute parallel for' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target teams distribute parallel for' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel // expected-error {{use of undeclared identifier 'parallel'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : argc / z)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp target teams distribute parallel for'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : argc) if (parallel:argc) // expected-error {{directive '#pragma omp target teams distribute parallel for' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(target: argc) if (target:argc) // expected-error {{directive '#pragma omp target teams distribute parallel for' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(parallel : argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{expected 'target' directive name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(target: argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{expected 'parallel' directive name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for if(distribute : argc) // expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute parallel for'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for default(none) if(argc+argv[0][0]) // expected-error {{variable 'argv' must have explicitly specified data sharing attributes}} expected-error {{variable 'argc' must have explicitly specified data sharing attributes}} expected-note 2 {{explicit data sharing attribute requested here}}
  for (i = 0; i < argc; ++i) foo();

  return tmain(argc, argv);
}
