// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -triple x86_64-unknown-unknown %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=61 -triple x86_64-unknown-unknown %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64 %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -triple x86_64-unknown-unknown -DVERSION52 %s

void foo() {
}

#ifndef VERSION52
void bar(int N) { // expected-note {{declared here}}
  // 1. Invalid syntax of the dims modifier.

#pragma omp target teams num_teams(dims 2: 4) // expected-error {{use of undeclared identifier 'dims'}}
  foo();

#pragma omp target thread_limit(dim(2) 4, 5)
  // expected-error@-1 {{use of undeclared identifier 'dim'}}
  // expected-error@-2 {{expected ',' or ')' in 'thread_limit' clause}}
  foo();

#pragma omp target thread_limit(dims((2): 4, 5)
  // expected-error@-1 {{expected ')'}}
  // expected-error@-2 {{expected ')'}}
  // expected-note@-3 {{to match this '('}}
  // expected-note@-4 {{to match this '('}}
  // expected-error@-5 {{missing ':' after thread_limit modifier}}
  foo();

#pragma omp target thread_limit(dims(2)): 4, 5)
  // expected-error@-1 {{missing ':' after thread_limit modifier}}
  // expected-warning@-2 {{extra tokens at the end of '#pragma omp target' are ignored}}
  foo();

#pragma omp target thread_limit(dims(2) 4, 5) // expected-error {{missing ':' after thread_limit modifier}}
  foo();

#pragma omp target teams distribute num_teams(dims(): 4) // expected-error {{expected expression}}
  for (int i = 0; i < 10; ++i) {}

  // 3. Incompatible modifiers.

#pragma omp target teams num_teams(dims(1),10:20) // expected-error {{'lower_bound' modifier cannot be specified with 'dims' modifier in 'num_teams' clause}}
  foo();

  // 2. Mismatching number of expressions.

#pragma omp target teams num_teams(dims(2): 4) // expected-error {{unexpected number of expressions in 'num_teams' clause}}
  foo();

#pragma omp target thread_limit(dims(1): 4, 5) // expected-error {{unexpected number of expressions in 'thread_limit' clause}}
  foo();

#pragma omp target teams distribute num_teams(dims(3): 4, 5) // expected-error {{unexpected number of expressions in 'num_teams' clause}}
  for (int i = 0; i < 10; ++i) {}

  // 3. Exceeding three dimensions.

#pragma omp target teams num_teams(dims(4): 1, 2, 3, 4) // expected-error {{maximum three expressions are supported in 'num_teams' clause}}
  foo();

#pragma omp target thread_limit(dims(2): 1, 2, 3, 4) // expected-error {{unexpected number of expressions in 'thread_limit' clause}}
  foo();

#pragma omp target teams distribute thread_limit(dims(4): 1, 2, 3, 4) // expected-error {{maximum three expressions are supported in 'thread_limit' clause}}
  for (int i = 0; i < 10; ++i) {}

  // 4. Invalid use of dims when ompx_bare is present.

#pragma omp target teams ompx_bare num_teams(dims(2): 1, 2) thread_limit(1, 2, 3) // expected-error {{'ompx_bare' clause cannot be specified with 'dims' modifier in 'num_teams' and 'thread_limit' clauses}}
  foo();

#pragma omp target teams ompx_bare num_teams(1, 2, 3) thread_limit(dims(3): 1, 2, 3) // expected-error {{'ompx_bare' clause cannot be specified with 'dims' modifier in 'num_teams' and 'thread_limit' clauses}}
  foo();

  // 5. Number of dimensions in dims is invalid.

#pragma omp target teams num_teams(dims(N): 1, 2)
  // expected-error@-1 {{expression is not an integral constant expression}}
  // expected-note@-2 {{function parameter 'N' with unknown value cannot be used in a constant expression}}
  foo();

#pragma omp target thread_limit(dims(2.5): 1, 2) // expected-error {{integral constant expression must have integral or unscoped enumeration type, not 'double'}}
  foo();

#pragma omp target teams distribute num_teams(dims(0): 4) // expected-error {{argument to 'num_teams' clause must be a strictly positive integer value}}
  for (int i = 0; i < 10; ++i) {}

#pragma omp target teams thread_limit(dims(-1): 4) // expected-error {{argument to 'thread_limit' clause must be a strictly positive integer value}}
  foo();
}

template <int D>
void template_test() {
  // 7. Mismatching number of expressions with template arguments.

#pragma omp target teams num_teams(dims(D): 4)
  // expected-error@-1 {{unexpected number of expressions in 'num_teams' clause}}
  // expected-error@-2 {{argument to 'num_teams' clause must be a strictly positive integer value}}
  foo();

#pragma omp target thread_limit(dims(D): 4, 5)
  // expected-error@-1 {{unexpected number of expressions in 'thread_limit' clause}}
  // expected-error@-2 {{argument to 'thread_limit' clause must be a strictly positive integer value}}
  foo();

#pragma omp target teams distribute num_teams(dims(D): 4, 5)
  // expected-error@-1 {{unexpected number of expressions in 'num_teams' clause}}
  // expected-error@-2 {{argument to 'num_teams' clause must be a strictly positive integer value}}
  for (int i = 0; i < 10; ++i) {}
}

void call_templates() {
  template_test<3>(); // expected-note {{in instantiation of function template specialization 'template_test<3>' requested here}}
  template_test<0>(); // expected-note {{in instantiation of function template specialization 'template_test<0>' requested here}}
}
#endif

#ifdef VERSION52
void version() {
  // 6. Dims modifier requires OpenMP 6.1.

#pragma omp target teams num_teams(dims(1): 4) // expected-error {{'dims' modifier in 'num_teams' clause requires OpenMP 6.1 or later}}
  foo();

#pragma omp target thread_limit(dims(1): 4) // expected-error {{'dims' modifier in 'thread_limit' clause requires OpenMP 6.1 or later}}
  foo();
}
#endif
