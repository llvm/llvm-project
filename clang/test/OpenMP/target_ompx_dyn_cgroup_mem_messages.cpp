// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  T z;
  #pragma omp target ompx_dyn_cgroup_mem // expected-error {{expected '(' after 'ompx_dyn_cgroup_mem'}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem () // expected-error {{expected expression}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (foobool(argc)), ompx_dyn_cgroup_mem (true) // expected-error {{directive '#pragma omp target' cannot contain more than one 'ompx_dyn_cgroup_mem' clause}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (S) // expected-error {{'S' does not refer to a value}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (argv[1]=2) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem(argc+z)
  foo();
  return 0;
}

int main(int argc, char **argv) {
int z;
  #pragma omp target ompx_dyn_cgroup_mem // expected-error {{expected '(' after 'ompx_dyn_cgroup_mem'}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem () // expected-error {{expected expression}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (foobool(argc)), ompx_dyn_cgroup_mem (true) // expected-error {{directive '#pragma omp target' cannot contain more than one 'ompx_dyn_cgroup_mem' clause}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (S1) // expected-error {{'S1' does not refer to a value}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (argv[1]=2) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target ompx_dyn_cgroup_mem(ompx_dyn_cgroup_mem(tmain(argc, argv) // expected-error2 {{expected ')'}} expected-note2 {{to match this '('}} expected-note {{in instantiation of function template specialization 'tmain<int, char>' requested here}}
  foo();

  return tmain(argc, argv);
}

