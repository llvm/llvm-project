// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=61 %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  T z;
  #pragma omp target teams dyn_groupprivate // expected-error {{expected '(' after 'dyn_groupprivate'}}
  foo();
  #pragma omp target teams dyn_groupprivate ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams dyn_groupprivate () // expected-error {{expected expression}}
  foo();
  #pragma omp target teams dyn_groupprivate (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams dyn_groupprivate (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams' are ignored}}
  foo();
  #pragma omp target teams dyn_groupprivate (argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target teams dyn_groupprivate (foobool(argc)), dyn_groupprivate (true) // expected-error {{directive '#pragma omp target teams' cannot contain more than one 'dyn_groupprivate' clause}}
  foo();
  #pragma omp target teams dyn_groupprivate (S) // expected-error {{'S' does not refer to a value}}
  foo();
  #pragma omp target teams dyn_groupprivate (argv[1]=2) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams dyn_groupprivate (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams dyn_groupprivate(argc+z)
  foo();
  return 0;
}

int main(int argc, char **argv) {
constexpr int n = -1;
int z;
  #pragma omp target teams dyn_groupprivate // expected-error {{expected '(' after 'dyn_groupprivate'}}
  foo();
  #pragma omp target teams dyn_groupprivate ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams dyn_groupprivate () // expected-error {{expected expression}}
  foo();
  #pragma omp target teams dyn_groupprivate (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams dyn_groupprivate (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams' are ignored}}
  foo();
  #pragma omp target teams dyn_groupprivate (argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target teams dyn_groupprivate (foobool(argc)), dyn_groupprivate (true) // expected-error {{directive '#pragma omp target teams' cannot contain more than one 'dyn_groupprivate' clause}}
  foo();
  #pragma omp target teams dyn_groupprivate (S1) // expected-error {{'S1' does not refer to a value}}
  foo();
  #pragma omp target teams dyn_groupprivate (argv[1]=2) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams dyn_groupprivate (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams dyn_groupprivate (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams dyn_groupprivate(dyn_groupprivate(tmain(argc, argv) // expected-error2 {{expected ')'}} expected-note2 {{to match this '('}} expected-note {{in instantiation of function template specialization 'tmain<int, char>' requested here}}
  foo();
  #pragma omp target teams dyn_groupprivate(-1) // expected-error {{argument to 'dyn_groupprivate' clause must be a non-negative integer value}}
  foo();
  #pragma omp target teams dyn_groupprivate(cgrou) // expected-error {{use of undeclared identifier 'cgrou'}}
  foo();
  #pragma omp target teams dyn_groupprivate(cgrou: argc) // expected-error {{use of undeclared identifier 'cgrou'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams dyn_groupprivate(cgroup,cgroup: argc) // expected-error {{modifier 'cgroup' cannot be used along with modifier 'cgroup' in dyn_groupprivate}}
  foo();
  #pragma omp target dyn_groupprivate(fallback(default_mem),fallback(abort): argc) // expected-error {{modifier 'fallback(abort)' cannot be used along with modifier 'fallback(default_mem)' in dyn_groupprivate}}
  foo();
  #pragma omp target dyn_groupprivate(fallback(abort),fallback(null): argc) // expected-error {{modifier 'fallback(null)' cannot be used along with modifier 'fallback(abort)' in dyn_groupprivate}}
  foo();
  #pragma omp target dyn_groupprivate(fallback(cgroup): argc) // expected-error {{expected 'abort', 'null' or 'default_mem' in fallback modifier}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target dyn_groupprivate(fallback(): argc) // expected-error {{expected 'abort', 'null' or 'default_mem' in fallback modifier}} expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams dyn_groupprivate(: argc) // expected-error {{expected ')'}} expected-error {{expected expression}} expected-note {{to match this '('}}
  foo();

  return tmain(argc, argv);
}

