// RUN: %clang_cc1 -verify -fopenmp %s

// RUN: %clang_cc1 -verify -fopenmp-simd %s

void foo() {
}

template <class T, typename S, int N, int ST>
T tmain(T argc, S **argv) {
  int i;
#pragma omp target teams distribute parallel for simd defaultmap // expected-error {{expected '(' after 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap ( // expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap () // expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom: // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom) // expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom scalar) // expected-warning {{missing ':' after defaultmap modifier - ignoring}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom, // expected-error {{expected ')'}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (scalar: // expected-error {{expected ')'}} expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom, scalar // expected-error {{expected ')'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

  return argc;
}

int main(int argc, char **argv) {
  int i;
#pragma omp target teams distribute parallel for simd defaultmap // expected-error {{expected '(' after 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap ( // expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap () // expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom: // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom) // expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom scalar) // expected-warning {{missing ':' after defaultmap modifier - ignoring}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom, // expected-error {{expected ')'}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (scalar: // expected-error {{expected ')'}} expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for simd defaultmap (tofrom, scalar // expected-error {{expected ')'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();

  return tmain<int, char, 1, 0>(argc, argv);
}

