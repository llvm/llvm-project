// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
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
#pragma omp target update to(n) if(argc)
#pragma omp target update from(n) if(target update // expected-warning {{missing ':' after directive name modifier - ignoring}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(n) if(target update : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update from(n) if(target update : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(n) if(target update : argc)
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
#pragma omp target update to(m) if(target update // expected-warning {{missing ':' after directive name modifier - ignoring}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update from(m) if(target update : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(m) if(target update : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update from(m) if(target update : argc)
#pragma omp target update to(m) if(target update : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp target update'}}
#pragma omp target update from(m) if(target update : argc) if (target update:argc)  // expected-error {{directive '#pragma omp target update' cannot contain more than one 'if' clause with 'target update' name modifier}}
#pragma omp target update to(m) if(target update : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}}
  return tmain(argc, argv);
}
