// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -ferror-limit 100 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  #pragma omp parallel
  {
    #pragma omp cancel parallel if // expected-error {{expected '(' after 'if'}}
    #pragma omp cancel parallel if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if () // expected-error {{expected expression}}
    #pragma omp cancel parallel if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp cancel' are ignored}}
    #pragma omp cancel parallel if (argc > 0 ? argv[1] : argv[2])
    #pragma omp cancel parallel if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp cancel' cannot contain more than one 'if' clause}}
    #pragma omp cancel parallel if (S) // expected-error {{'S' does not refer to a value}}
    #pragma omp cancel parallel if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if(argc)
    #pragma omp cancel parallel if(cancel // expected-error {{use of undeclared identifier 'cancel'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if(cancel : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if(cancel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if(cancel : argc)
    #pragma omp cancel parallel if(cancel : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp cancel'}}
    #pragma omp cancel parallel if(cancel : argc) if (cancel:argc) // expected-error {{directive '#pragma omp cancel' cannot contain more than one 'if' clause with 'cancel' name modifier}}
    #pragma omp cancel parallel if(cancel : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}}
    foo();
  }

  return 0;
}

int main(int argc, char **argv) {
  #pragma omp parallel
  {
    #pragma omp cancel parallel if // expected-error {{expected '(' after 'if'}}
    #pragma omp cancel parallel if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if () // expected-error {{expected expression}}
    #pragma omp cancel parallel if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp cancel' are ignored}}
    #pragma omp cancel parallel if (argc > 0 ? argv[1] : argv[2])
    #pragma omp cancel parallel if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp cancel' cannot contain more than one 'if' clause}}
    #pragma omp cancel parallel if (S1) // expected-error {{'S1' does not refer to a value}}
    #pragma omp cancel parallel if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if(cancel // expected-error {{use of undeclared identifier 'cancel'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if(cancel : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if(cancel : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
    #pragma omp cancel parallel if(cancel : argc)
    #pragma omp cancel parallel if(cancel : argc) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp cancel'}}
    #pragma omp cancel parallel if(cancel : argc) if (cancel:argc)  // expected-error {{directive '#pragma omp cancel' cannot contain more than one 'if' clause with 'cancel' name modifier}}
    #pragma omp cancel parallel if(cancel : argc) if (argc) // expected-error {{no more 'if' clause is allowed}} expected-note {{previous clause with directive name modifier specified here}}
    foo();
  }

  return tmain(argc, argv);
}
