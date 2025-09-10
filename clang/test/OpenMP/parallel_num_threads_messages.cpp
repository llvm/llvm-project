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
  #pragma omp parallel num_threads // expected-error {{expected '(' after 'num_threads'}}
  #pragma omp parallel num_threads ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel num_threads () // expected-error {{expected expression}}
  #pragma omp parallel num_threads (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel num_threads (argc)) // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  #pragma omp parallel num_threads ((argc > 0) ? argv[1] : argv[2]) // expected-error 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  #pragma omp parallel num_threads (foobool(argc)), num_threads (true), num_threads (-5) // expected-error 2 {{directive '#pragma omp parallel' cannot contain more than one 'num_threads' clause}} expected-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  #pragma omp parallel num_threads (S) // expected-error {{'S' does not refer to a value}}
  #pragma omp parallel num_threads (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  #pragma omp parallel num_threads (argc+z)
  #pragma omp parallel num_threads (N) // expected-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  #pragma omp parallel redef_num_threads (argc, argc)

#ifdef OMP60
  // Valid uses of strict modifier
  #pragma omp parallel num_threads(strict: 4)
  #pragma omp parallel num_threads(strict: argc+z)

  // Invalid: missing expression after strict:
  #pragma omp parallel num_threads(strict: ) // omp60-error {{expected expression}}
  #pragma omp parallel num_threads(strict:) // omp60-error {{expected expression}}
  #pragma omp parallel num_threads(strict: // omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}

  // Invalid: unknown/missing modifier
  #pragma omp parallel num_threads(foo: 4) // omp60-error {{expected 'strict' in OpenMP clause 'num_threads'}}
  #pragma omp parallel num_threads(: 4) // omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}
  #pragma omp parallel num_threads(:)// omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}

  // Invalid: missing colon after modifier
  #pragma omp parallel num_threads(strict 4) // omp60-error {{missing ':' after strict modifier}}

  // Invalid: negative, zero, or non-integral
  #pragma omp parallel num_threads(strict: -1) // omp60-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  #pragma omp parallel num_threads(strict: 0) // omp60-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  #pragma omp parallel num_threads(strict: (argc > 0) ? argv[1] : argv[2]) // omp60-error 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  #pragma omp parallel num_threads(strict: S) // omp60-error {{'S' does not refer to a value}}
  #pragma omp parallel num_threads(strict: argv[1]=2) // omp60-error {{expected ')'}} omp60-note {{to match this '('}} omp60-error 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  #pragma omp parallel num_threads(strict: N) // omp60-error {{argument to 'num_threads' clause must be a strictly positive integer value}}

  // Invalid: multiple strict modifiers or mixed with non-strict
  #pragma omp parallel num_threads(strict: 4, strict: 5) // omp60-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel num_threads(strict: 4), num_threads(5) // omp60-error {{directive '#pragma omp parallel' cannot contain more than one 'num_threads' clause}}
  #pragma omp parallel num_threads(4), num_threads(strict: 5)  // omp60-error {{directive '#pragma omp parallel' cannot contain more than one 'num_threads' clause}}
#endif // OMP60

  foo();

  return argc;
}

int main(int argc, char **argv) {
int z;
  #pragma omp parallel num_threads // expected-error {{expected '(' after 'num_threads'}}
  #pragma omp parallel num_threads ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel num_threads () // expected-error {{expected expression}}
  #pragma omp parallel num_threads (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel num_threads (z+argc)) // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  #pragma omp parallel num_threads (argc > 0 ? argv[1] : argv[2]) // expected-error {{integral }}
  #pragma omp parallel num_threads (foobool(argc)), num_threads (true), num_threads (-5) // expected-error 2 {{directive '#pragma omp parallel' cannot contain more than one 'num_threads' clause}} expected-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  #pragma omp parallel num_threads (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp parallel num_threads (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  #pragma omp parallel num_threads (num_threads(tmain<int, char, -1>(argc, argv) // expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}} expected-note {{in instantiation of function template specialization 'tmain<int, char, -1>' requested here}}
  #pragma omp parallel redef_num_threads (argc, argc)

#ifdef OMP60
  // Valid uses of strict modifier
  #pragma omp parallel num_threads(strict: 4)
  #pragma omp parallel num_threads(strict: argc+z)

  // Invalid: missing expression after strict:
  #pragma omp parallel num_threads(strict: ) // omp60-error {{expected expression}}
  #pragma omp parallel num_threads(strict:) // omp60-error {{expected expression}}
  #pragma omp parallel num_threads(strict: // omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}

  // Invalid: unknown/missing modifier
  #pragma omp parallel num_threads(foo: 4) // omp60-error {{expected 'strict' in OpenMP clause 'num_threads'}}
  #pragma omp parallel num_threads(: 4) // omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}
  #pragma omp parallel num_threads(:) // omp60-error {{expected expression}} omp60-error {{expected ')'}} omp60-note {{to match this '('}}

  // Invalid: missing colon after modifier
  #pragma omp parallel num_threads(strict 4) // omp60-error {{missing ':' after strict modifier}}

  // Invalid: negative, zero, or non-integral
  #pragma omp parallel num_threads(strict: -1) // omp60-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  #pragma omp parallel num_threads(strict: 0) // omp60-error {{argument to 'num_threads' clause must be a strictly positive integer value}}
  #pragma omp parallel num_threads(strict: (argc > 0) ? argv[1] : argv[2]) // omp60-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  #pragma omp parallel num_threads(strict: S1) // omp60-error {{'S1' does not refer to a value}}
  #pragma omp parallel num_threads(strict: argv[1]=2) // omp60-error {{expected ')'}} omp60-note {{to match this '('}} omp60-error {{expression must have integral or unscoped enumeration type, not 'char *'}}

  // Invalid: multiple strict modifiers or mixed with non-strict
  #pragma omp parallel num_threads(strict: 4, strict: 5) // omp60-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel num_threads(strict: 4), num_threads(5) // omp60-error {{directive '#pragma omp parallel' cannot contain more than one 'num_threads' clause}}
  #pragma omp parallel num_threads(4), num_threads(strict: 5) // omp60-error {{directive '#pragma omp parallel' cannot contain more than one 'num_threads' clause}}
#endif // OMP60

  foo();

  return tmain<int, char, 3>(argc, argv); // expected-note {{in instantiation of function template specialization 'tmain<int, char, 3>' requested here}}
}
