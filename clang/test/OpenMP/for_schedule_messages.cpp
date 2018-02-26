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
  #pragma omp for schedule // expected-error {{expected '(' after 'schedule'}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule ( // expected-error {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule () // expected-error {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (auto // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (monotonic // expected-error {{expected ')'}} expected-error {{expected 'static', 'dynamic', 'guided', 'auto' or 'runtime' in OpenMP clause 'schedule'}} expected-warning {{missing ':' after schedule modifier - ignoring}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (auto_dynamic // expected-error {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (nonmonotonic, // expected-error {{expected ')'}} expected-error {{expected 'simd' in OpenMP clause 'schedule'}} expected-warning {{missing ':' after schedule modifier - ignoring}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (simd: // expected-error {{expected ')'}} expected-error {{expected 'static', 'dynamic', 'guided', 'auto' or 'runtime' in OpenMP clause 'schedule'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (monotonic, auto // expected-error {{expected ')'}} expected-warning {{missing ':' after schedule modifier - ignoring}} expected-error {{expected 'simd' in OpenMP clause 'schedule'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (nonmonotonic: auto, // expected-error {{expected ')'}} expected-error {{'nonmonotonic' modifier can only be specified with 'dynamic' or 'guided' schedule kind}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (auto,  // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (runtime, 3)  // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  // expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
  #pragma omp for schedule (guided argc
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  // expected-error@+1 2 {{argument to 'schedule' clause must be a strictly positive integer value}}
  #pragma omp for schedule (static, ST // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (dynamic, 1)) // expected-warning {{extra tokens at the end of '#pragma omp for' are ignored}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (guided, (ST > 0) ? 1 + ST : 2)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  // expected-error@+2 2 {{directive '#pragma omp for' cannot contain more than one 'schedule' clause}}
  // expected-error@+1 {{argument to 'schedule' clause must be a strictly positive integer value}}
  #pragma omp for schedule (static, foobool(argc)), schedule (dynamic, true), schedule (guided, -5)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (static, S) // expected-error {{'S' does not refer to a value}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  // expected-error@+1 2 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  #pragma omp for schedule (guided, argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (dynamic, 1)
  for (int i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (static, N) // expected-error {{argument to 'schedule' clause must be a strictly positive integer value}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (nonmonotonic: static) // expected-error {{'nonmonotonic' modifier can only be specified with 'dynamic' or 'guided' schedule kind}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (nonmonotonic: auto) // expected-error {{'nonmonotonic' modifier can only be specified with 'dynamic' or 'guided' schedule kind}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (nonmonotonic: runtime) // expected-error {{'nonmonotonic' modifier can only be specified with 'dynamic' or 'guided' schedule kind}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (monotonic, nonmonotonic: auto) // expected-error {{modifier 'nonmonotonic' cannot be used along with modifier 'monotonic'}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (nonmonotonic, monotonic: auto) // expected-error {{modifier 'monotonic' cannot be used along with modifier 'nonmonotonic'}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (nonmonotonic, nonmonotonic: auto) // expected-error {{modifier 'nonmonotonic' cannot be used along with modifier 'nonmonotonic'}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (monotonic, monotonic: auto) // expected-error {{modifier 'monotonic' cannot be used along with modifier 'monotonic'}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for schedule (nonmonotonic: guided) ordered // expected-error {{'schedule' clause with 'nonmonotonic' modifier cannot be specified if an 'ordered' clause is specified}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  #pragma omp for ordered(1) schedule(nonmonotonic: dynamic) // expected-error {{'schedule' clause with 'nonmonotonic' modifier cannot be specified if an 'ordered' clause is specified}}
  for (T i = ST; i < N; i++) argv[0][i] = argv[0][i] - argv[0][i-ST];
  return argc;
}

int main(int argc, char **argv) {
  #pragma omp for schedule // expected-error {{expected '(' after 'schedule'}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  #pragma omp for schedule ( // expected-error {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  #pragma omp for schedule () // expected-error {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  #pragma omp for schedule (auto // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  #pragma omp for schedule (auto_dynamic // expected-error {{expected 'static', 'dynamic', 'guided', 'auto', 'runtime', 'monotonic', 'nonmonotonic' or 'simd' in OpenMP clause 'schedule'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  #pragma omp for schedule (auto,  // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  #pragma omp for schedule (runtime, 3)  // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  #pragma omp for schedule (guided, 4 // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  #pragma omp for schedule (static, 2+2)) // expected-warning {{extra tokens at the end of '#pragma omp for' are ignored}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  #pragma omp for schedule (dynamic, foobool(1) > 0 ? 1 : 2)
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  // expected-error@+2 2 {{directive '#pragma omp for' cannot contain more than one 'schedule' clause}}
  // expected-error@+1 {{argument to 'schedule' clause must be a strictly positive integer value}}
  #pragma omp for schedule (guided, foobool(argc)), schedule (static, true), schedule (dynamic, -5)
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  #pragma omp for schedule (guided, S1) // expected-error {{'S1' does not refer to a value}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  // expected-error@+1 {{expression must have integral or unscoped enumeration type, not 'char *'}}
  #pragma omp for schedule (static, argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++) argv[0][i] = argv[0][i] - argv[0][i-4];
  // expected-error@+3 {{statement after '#pragma omp for' must be a for loop}}
  // expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, -1, -2>' requested here}}
  #pragma omp for schedule(dynamic, schedule(tmain<int, char, -1, -2>(argc, argv) // expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}}
  foo();
  // expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, 1, 0>' requested here}}
  return tmain<int, char, 1, 0>(argc, argv);
}

