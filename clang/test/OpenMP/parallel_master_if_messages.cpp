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
#pragma omp parallel master if(cond) // expected-warning {{variable 'cond' is uninitialized when used here}}
  {
    ;
  }
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  T z;
  #pragma omp parallel master if // expected-error {{expected '(' after 'if'}}
  {
    foo();
  }
  #pragma omp parallel master if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if () // expected-error {{expected expression}}
  {
    foo();
  }
  #pragma omp parallel master if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp parallel master' are ignored}}
  {
    foo();
  }
  #pragma omp parallel master if (argc > 0 ? argv[1] : argv[2])
  {
    foo();
  }
  #pragma omp parallel master if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp parallel master' cannot contain more than one 'if' clause}}
  {
    foo();
  }
  #pragma omp parallel master if (S) // expected-error {{'S' does not refer to a value}}
  {
    foo();
  }
  #pragma omp parallel master if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if(argc + z)
  {
    foo();
  }
  #pragma omp parallel master if(parallel : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if(parallel : argc)
  {
    foo();
  }
  #pragma omp parallel master if(parallel : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp parallel master'}}
  {
    foo();
  }
  #pragma omp parallel master if(parallel : argc) if (parallel:argc) // expected-error {{directive '#pragma omp parallel master' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  {
    foo();
  }
  #pragma omp parallel master if(parallel : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}}
  {
    foo();
  }

  return 0;
}

int main(int argc, char **argv) {
  int z;
  #pragma omp parallel master if // expected-error {{expected '(' after 'if'}}
  {
    foo();
  }
  #pragma omp parallel master if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if () // expected-error {{expected expression}}
  {
    foo();
  }
  #pragma omp parallel master if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp parallel master' are ignored}}
  {
    foo();
  }
  #pragma omp parallel master if (argc > 0 ? argv[1] : argv[2] + z)
  {
    foo();
  }
  #pragma omp parallel master if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp parallel master' cannot contain more than one 'if' clause}}
  {
    foo();
  }
  #pragma omp parallel master if (S1) // expected-error {{'S1' does not refer to a value}}
  {
    foo();
  }
  #pragma omp parallel master if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if(parallel : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if(parallel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  {
    foo();
  }
  #pragma omp parallel master if(parallel : argc)
  {
    foo();
  }
  #pragma omp parallel master if(parallel : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp parallel master'}}
  {
    foo();
  }
  #pragma omp parallel master if(parallel : argc) if (parallel:argc) // expected-error {{directive '#pragma omp parallel master' cannot contain more than one 'if' clause with 'parallel' name modifier}}
  {
    foo();
  }
  #pragma omp parallel master if(parallel : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}}
  {
    foo();
  }

  return tmain(argc, argv);
}
