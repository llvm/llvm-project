// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 %s -verify=expected,omp5 -Wuninitialized -DOMP5
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 %s -verify=expected,omp5 -Wuninitialized -DOMP5

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 %s -verify=expected,omp45 -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 %s -verify=expected,omp45 -Wuninitialized

void foo() {
}

template <class T, typename S, int N, int ST>
T tmain(T argc, S **argv) {
  int i;
  #pragma omp target simd defaultmap // expected-error {{expected '(' after 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target simd defaultmap( // omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target simd defaultmap() // omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target simd defaultmap(tofrom // expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target simd defaultmap(tofrom: // expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap(tofrom) // omp45-warning {{missing ':' after defaultmap modifier - ignoring}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap(tofrom scalar) // omp45-warning {{missing ':' after defaultmap modifier - ignoring}} omp5-error {{expected ')'}} omp5-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap(tofrom, // expected-error {{expected ')'}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp target simd defaultmap (scalar: // expected-error {{expected ')'}} omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap(tofrom, scalar // expected-error {{expected ')'}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();

  return argc;
}

int main(int argc, char **argv) {
  int i;
#pragma omp target simd defaultmap // expected-error {{expected '(' after 'defaultmap'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap( // omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap() // omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap(tofrom // expected-error {{expected ')'}} expected-note {{to match this '('}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap(tofrom: // expected-error {{expected ')'}} expected-note {{to match this '('}} omp5-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap(tofrom) // omp45-warning {{missing ':' after defaultmap modifier - ignoring}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap(tofrom scalar) // omp45-warning {{missing ':' after defaultmap modifier - ignoring}} omp5-error {{expected ')'}} omp5-note {{to match this '('}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap(tofrom, // expected-error {{expected ')'}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap(scalar: // expected-error {{expected ')'}} omp5-error {{expected 'alloc', 'from', 'to', 'tofrom', 'firstprivate', 'none', 'default' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}} omp45-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}} omp5-error {{expected 'scalar', 'aggregate', 'pointer' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target simd defaultmap(tofrom, scalar // expected-error {{expected ')'}} omp45-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}} omp45-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  for (i = 0; i < argc; ++i) foo();

  return tmain<int, char, 1, 0>(argc, argv);
}
