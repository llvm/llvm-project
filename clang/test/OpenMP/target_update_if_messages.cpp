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
#pragma omp target update to(argc) if(cond) // expected-warning {{variable 'cond' is uninitialized when used here}}
  for (int i = 0; i < 10; ++i)
    ;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  int n;
#pragma omp target update to(n) if // expected-error {{expected '(' after 'if'}}
#pragma omp target update from(n) if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(n) if () // expected-error {{expected expression}}
#pragma omp target update from(n) if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(n) if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
#pragma omp target update from(n) if (argc > 0 ? argv[1] : argv[2])
#pragma omp target update to(n) if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target update' cannot contain more than one 'if' clause}}
#pragma omp target update from(n) if (S) // expected-error {{'S' does not refer to a value}}
#pragma omp target update to(n) if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update from(n) if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(n) if(argc + n)
#pragma omp target update from(n) if(target update // expected-error {{use of undeclared identifier 'target'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(n) if(target update : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update from(n) if(target update : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(n) if(target update : argc + n)
#pragma omp target update from(n) if(target update : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp target update'}}
#pragma omp target update to(n) if(target update : argc) if (target update:argc) // expected-error {{directive '#pragma omp target update' cannot contain more than one 'if' clause with 'target update' name modifier}}
#pragma omp target update from(n) if(target update : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}}
  return 0;
}

int main(int argc, char **argv) {
  int m;
#pragma omp target update to(m) if // expected-error {{expected '(' after 'if'}}
#pragma omp target update from(m) if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(m) if () // expected-error {{expected expression}}
#pragma omp target update from(m) if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(m) if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
#pragma omp target update from(m) if (argc > 0 ? argv[1] : argv[2])
#pragma omp target update to(m) if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target update' cannot contain more than one 'if' clause}}
#pragma omp target update from(m) if (S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp target update to(m) if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update from(m) if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(m) if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update from(m) if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(m) if(target update // expected-error {{use of undeclared identifier 'target'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update from(m) if(target update : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(m) if(target update : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update from(m) if(target update : argc + m)
#pragma omp target update to(m) if(target update : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp target update'}}
#pragma omp target update from(m) if(target update : argc) if (target update:argc)  // expected-error {{directive '#pragma omp target update' cannot contain more than one 'if' clause with 'target update' name modifier}}
#pragma omp target update to(m) if(target update : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}}
  return tmain(argc, argv);
}
