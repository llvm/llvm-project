// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -DOMP60 -verify=expected,omp60 -fopenmp -fopenmp-version=60 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -DOMP60 -verify=expected,omp60 -fopenmp-simd -fopenmp-version=60 -ferror-limit 100 %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}} omp60-note {{declared here}}

#define redef_num_threads(a, b) num_threads(a)

template <class T, typename S, int N> // expected-note {{declared here}} omp60-note {{declared here}}
T tmain(T argc, S **argv) {
  T z;
  #pragma omp target parallel num_threads // expected-error {{expected '(' after 'num_threads'}}
  foo();
  #pragma omp target parallel num_threads ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads () // expected-error {{expected expression}}
  foo();
  #pragma omp target parallel num_threads (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel num_threads ((argc > 0) ? argv[1] : argv[2]) // expected-error 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target parallel num_threads (foobool(argc)), num_threads (true), num_threads (-5) // expected-error 2 {{directive '#pragma omp target parallel' cannot contain more than one 'num_threads' clause}} expected-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  foo();
  #pragma omp target parallel num_threads (S) // expected-error {{'S' does not refer to a value}}
  foo();
  #pragma omp target parallel num_threads (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target parallel num_threads (argc + z)
  foo();
  #pragma omp target parallel num_threads (N) // expected-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  foo();
  #pragma omp target parallel redef_num_threads (argc, argc)
  foo();

#ifdef OMP60
  // Valid uses of strict modifier
  #pragma omp target parallel num_threads(strict: 4)
  foo();
  #pragma omp target parallel num_threads(strict: argc+z)
  foo();

  // Invalid: missing expression after strict:
  #pragma omp target parallel num_threads(strict: ) // omp60-error {{expected expression}}
  foo();
  #pragma omp target parallel num_threads(strict:) // omp60-error {{expected expression}}
  foo();
  #pragma omp target parallel num_threads(strict: // omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}
  foo();

  // Invalid: unknown/missing modifier
  #pragma omp target parallel num_threads(foo: 4) // omp60-error {{expected 'strict' in OpenMP clause 'num_threads'}}
  foo();
  #pragma omp target parallel num_threads(: 4) // omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads(:)// omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}
  foo();

  // Invalid: missing colon after modifier
  #pragma omp target parallel num_threads(strict 4) // omp60-error {{missing ':' after strict modifier}}
  foo();

  // Invalid: negative, zero, or non-integral
  #pragma omp target parallel num_threads(strict: -1) // omp60-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  foo();
  #pragma omp target parallel num_threads(strict: 0) // omp60-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  foo();
  #pragma omp target parallel num_threads(strict: (argc > 0) ? argv[1] : argv[2]) // omp60-error 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target parallel num_threads(strict: S) // omp60-error {{'S' does not refer to a value}}
  foo();
  #pragma omp target parallel num_threads(strict: argv[1]=2) // omp60-error {{expected ')'}} omp60-note {{to match this '('}} omp60-error 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target parallel num_threads(strict: N) // omp60-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  foo();

  // Invalid: multiple strict modifiers or mixed with non-strict
  #pragma omp target parallel num_threads(strict: 4, strict: 5) // omp60-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads(strict: 4), num_threads(5) // omp60-error {{directive '#pragma omp target parallel' cannot contain more than one 'num_threads' clause}}
  foo();
  #pragma omp target parallel num_threads(4), num_threads(strict: 5)  // omp60-error {{directive '#pragma omp target parallel' cannot contain more than one 'num_threads' clause}}
  foo();
#endif // OMP60

  return argc;
}

int main(int argc, char **argv) {
  int z;
  #pragma omp target parallel num_threads // expected-error {{expected '(' after 'num_threads'}}
  foo();
  #pragma omp target parallel num_threads ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads () // expected-error {{expected expression}}
  foo();
  #pragma omp target parallel num_threads (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads (z + argc)) // expected-warning {{extra tokens at the end of '#pragma omp target parallel' are ignored}}
  foo();
  #pragma omp target parallel num_threads (argc > 0 ? argv[1] : argv[2]) // expected-error {{integral }}
  foo();
  #pragma omp target parallel num_threads (foobool(argc)), num_threads (true), num_threads (-5) // expected-error 2 {{directive '#pragma omp target parallel' cannot contain more than one 'num_threads' clause}} expected-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  foo();
  #pragma omp target parallel num_threads (S1) // expected-error {{'S1' does not refer to a value}}
  foo();
  #pragma omp target parallel num_threads (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target parallel num_threads (num_threads(tmain<int, char, -1>(argc, argv) // expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}} expected-note {{in instantiation of function template specialization 'tmain<int, char, -1>' requested here}}
  foo();
  #pragma omp target parallel redef_num_threads (argc, argc)
  foo();

#ifdef OMP60
  // Valid uses of strict modifier
  #pragma omp target parallel num_threads(strict: 4)
  foo();
  #pragma omp target parallel num_threads(strict: argc+z)
  foo();

  // Invalid: missing expression after strict:
  #pragma omp target parallel num_threads(strict: ) // omp60-error {{expected expression}}
  foo();
  #pragma omp target parallel num_threads(strict:) // omp60-error {{expected expression}}
  foo();
  #pragma omp target parallel num_threads(strict: // omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}
  foo();

  // Invalid: unknown/missing modifier
  #pragma omp target parallel num_threads(foo: 4) // omp60-error {{expected 'strict' in OpenMP clause 'num_threads'}}
  foo();
  #pragma omp target parallel num_threads(: 4) // omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads(:) // omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}
  foo();

  // Invalid: missing colon after modifier
  #pragma omp target parallel num_threads(strict 4) // omp60-error {{missing ':' after strict modifier}}
  foo();

  // Invalid: negative, zero, or non-integral
  #pragma omp target parallel num_threads(strict: -1) // omp60-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  foo();
  #pragma omp target parallel num_threads(strict: 0) // omp60-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  foo();
  #pragma omp target parallel num_threads(strict: (argc > 0) ? argv[1] : argv[2]) // omp60-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target parallel num_threads(strict: S1) // omp60-error {{'S1' does not refer to a value}}
  foo();
  #pragma omp target parallel num_threads(strict: argv[1]=2) // omp60-error {{expected ')'}} omp60-note {{to match this '('}} omp60-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();

  // Invalid: multiple strict modifiers or mixed with non-strict
  #pragma omp target parallel num_threads(strict: 4, strict: 5) // omp60-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target parallel num_threads(strict: 4), num_threads(5) // omp60-error {{directive '#pragma omp target parallel' cannot contain more than one 'num_threads' clause}}
  foo();
  #pragma omp target parallel num_threads(4), num_threads(strict: 5) // omp60-error {{directive '#pragma omp target parallel' cannot contain more than one 'num_threads' clause}}
  foo();
#endif // OMP60

  return tmain<int, char, 3>(argc, argv); // expected-note {{in instantiation of function template specialization 'tmain<int, char, 3>' requested here}}
}
