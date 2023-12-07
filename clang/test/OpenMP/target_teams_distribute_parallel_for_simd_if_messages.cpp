// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp -fopenmp-version=45 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-simd -fopenmp-version=45 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

void xxx(int argc) {
  int cond; // expected-note {{initialize the variable 'cond' to silence this warning}}
#pragma omp target teams distribute parallel for simd if (parallel \
                                                          : cond) // expected-warning {{variable 'cond' is uninitialized when used here}}
  for (int i = 0; i < 10; ++i)
    ;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  T i, k;
#pragma omp target teams distribute parallel for simd if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (k)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute parallel for simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target teams distribute parallel for simd' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (S) // expected-error {{'S' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(argc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel // expected-error {{use of undeclared identifier 'parallel'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target // expected-error {{use of undeclared identifier 'target'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : argc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target: argc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : argc) if (simd:argc) // omp45-error {{directive name modifier 'simd' is not allowed for '#pragma omp target teams distribute parallel for simd'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : argc) if (parallel:argc) // expected-error {{directive '#pragma omp target teams distribute parallel for simd' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target: argc) if (target:argc) // expected-error {{directive '#pragma omp target teams distribute parallel for simd' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} omp45-error {{expected 'target' directive name modifier}} omp51-error {{expected one of 'target' or 'simd' directive name modifiers}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target: argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} omp45-error {{expected 'parallel' directive name modifier}} omp51-error {{expected one of 'parallel' or 'simd' directive name modifiers}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(distribute : argc) // expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute parallel for simd'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target: argc) if(parallel:argc) // OK
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel: argc) if(target:argc) // OK
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

int main(int argc, char **argv) {
  int i, k;
#pragma omp target teams distribute parallel for simd if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (k)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute parallel for simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target teams distribute parallel for simd' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel // expected-error {{use of undeclared identifier 'parallel'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target // expected-error {{use of undeclared identifier 'target'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target: argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : argc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target: argc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : argc) if (simd:argc) // omp45-error {{directive name modifier 'simd' is not allowed for '#pragma omp target teams distribute parallel for simd'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : argc) if (parallel:argc) // expected-error {{directive '#pragma omp target teams distribute parallel for simd' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target: argc) if (target:argc) // expected-error {{directive '#pragma omp target teams distribute parallel for simd' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel : argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} omp45-error {{expected 'target' directive name modifier}} omp51-error {{expected one of 'target' or 'simd' directive name modifiers}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target: argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} omp45-error {{expected 'parallel' directive name modifier}} omp51-error {{expected one of 'parallel' or 'simd' directive name modifiers}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(distribute : argc) // expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute parallel for simd'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(target: argc) if(parallel:argc) // OK
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd if(parallel: argc) if(target:argc) // OK
  for (i = 0; i < argc; ++i) foo();

  return tmain(argc, argv);
}
