// RUN: %clang_cc1 -verify -fopenmp %s

// RUN: %clang_cc1 -verify -fopenmp-simd %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, typename S, int N, int ST> // expected-note {{declared here}}
T tmain(T argc, S **argv) {
// expected-error@+1 {{expected '(' after 'schedule'}}
#pragma omp target teams distribute parallel for simd schedule
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+1 {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+1 {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}}
#pragma omp target teams distribute parallel for simd schedule ()
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (auto
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+1 {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (auto_dynamic
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (auto,
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (runtime, 3)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (guided argc
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+1 2 {{argument to 'schedule' clause must be a strictly positive integer value}}
#pragma omp target teams distribute parallel for simd schedule (static, ST // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-warning@+1 {{extra tokens at the end of '#pragma omp target teams distribute parallel for simd' are ignored}}
#pragma omp target teams distribute parallel for simd schedule (dynamic, 1))
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target teams distribute parallel for simd schedule (guided, (ST > 0) ? 1 + ST : 2)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+2 2 {{directive '#pragma omp target teams distribute parallel for simd' cannot contain more than one 'schedule' clause}}
// expected-error@+1 {{argument to 'schedule' clause must be a strictly positive integer value}}
#pragma omp target teams distribute parallel for simd schedule (static, foobool(argc)), schedule (dynamic, true), schedule (guided, -5)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+1 {{'S' does not refer to a value}}
#pragma omp target teams distribute parallel for simd schedule (static, S)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+2 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (guided, argv[1]=2)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target teams distribute parallel for simd schedule (dynamic, 1)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+1 {{argument to 'schedule' clause must be a strictly positive integer value}}
#pragma omp target teams distribute parallel for simd schedule (static, N)
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];

  return argc;
}

int main(int argc, char **argv) {
// expected-error@+1 {{expected '(' after 'schedule'}}
#pragma omp target teams distribute parallel for simd schedule
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+1 {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+1 {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}}
#pragma omp target teams distribute parallel for simd schedule ()
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (auto
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+1 {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (auto_dynamic
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (auto,
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (runtime, 3)
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute parallel for simd schedule (guided, 4
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-warning@+1 {{extra tokens at the end of '#pragma omp target teams distribute parallel for simd' are ignored}}
#pragma omp target teams distribute parallel for simd schedule (static, 2+2))
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target teams distribute parallel for simd schedule (dynamic, foobool(1) > 0 ? 1 : 2)
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+2 2 {{directive '#pragma omp target teams distribute parallel for simd' cannot contain more than one 'schedule' clause}}
// expected-error@+1 {{argument to 'schedule' clause must be a strictly positive integer value}}
#pragma omp target teams distribute parallel for simd schedule (guided, foobool(argc)), schedule (static, true), schedule (dynamic, -5)
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+1 {{'S1' does not refer to a value}}
#pragma omp target teams distribute parallel for simd schedule (guided, S1)
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expression must have integral or unscoped enumeration type, not 'char *'}}
#pragma omp target teams distribute parallel for simd schedule (static, argv[1]=2)
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];

// expected-error@+3 {{statement after '#pragma omp target teams distribute parallel for simd' must be a for loop}}
// expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, -1, -2>' requested here}}
#pragma omp target teams distribute parallel for simd schedule(dynamic, schedule(tmain<int, char, -1, -2>(argc, argv) // expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}}
  foo();

  // expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, 1, 0>' requested here}}
  return tmain<int, char, 1, 0>(argc, argv);
}

