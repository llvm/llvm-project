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
  #pragma omp target simd if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target simd' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (S) // expected-error {{'S' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(argc)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(target : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(target : argc)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(target : argc) if (simd:argc) // expected-error {{directive name modifier 'simd' is not allowed for '#pragma omp target simd'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(target : argc) if (target :argc) // expected-error {{directive '#pragma omp target simd' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(target : argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{no more 'if' clause is allowed}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

int main(int argc, char **argv) {
  int i;
  #pragma omp target simd if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target simd' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(target : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(target : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(target : argc) if (simd:argc) // expected-error {{directive name modifier 'simd' is not allowed for '#pragma omp target simd'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(target : argc) if (target :argc) // expected-error {{directive '#pragma omp target simd' cannot contain more than one 'if' clause with 'target' name modifier}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd if(target : argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{no more 'if' clause is allowed}}
  for (i = 0; i < argc; ++i) foo();

  return tmain(argc, argv);
}
