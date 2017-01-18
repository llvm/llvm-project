// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ferror-limit 100 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  T i;
#pragma omp target teams distribute simd if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target teams distribute simd' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (S) // expected-error {{'S' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(argc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target // expected-error {{use of undeclared identifier 'target'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target: argc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target : argc) if (distribute:argc) // expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute simd'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target: argc) if (target:argc) // expected-error {{directive '#pragma omp target teams distribute simd' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target: argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{no more 'if' clause is allowed}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(distribute : argc) // expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute simd'}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

int main(int argc, char **argv) {
  int i;
#pragma omp target teams distribute simd if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target teams distribute simd' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target // expected-error {{use of undeclared identifier 'target'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target: argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target: argc)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target : argc) if (distribute:argc) // expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute simd'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target: argc) if (target:argc) // expected-error {{directive '#pragma omp target teams distribute simd' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(target: argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{no more 'if' clause is allowed}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd if(distribute : argc) // expected-error {{directive name modifier 'distribute' is not allowed for '#pragma omp target teams distribute simd'}}
  for (i = 0; i < argc; ++i) foo();

  return tmain(argc, argv);
}
