// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ferror-limit 150 -o - %s

int incomplete[];

void test(int *p) {
  int a;
#pragma omp parallel reduction( // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-warning {{missing ':' after reduction identifier - ignoring}} expected-note {{to match this '('}}
  ;
#pragma omp parallel reduction(unknown // expected-error {{expected expression}} expected-error {{expected ')'}} expected-warning {{missing ':' after reduction identifier - ignoring}} expected-note {{to match this '('}}
  ;
#pragma omp parallel reduction(default, // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-warning {{missing ':' after reduction identifier - ignoring}} expected-note {{to match this '('}}
  ;
#pragma omp parallel reduction(unknown, +: a) // expected-error {{expected 'default', 'inscan' or 'task' in OpenMP clause 'reduction'}}
  ;
#pragma omp parallel reduction(default, + : a)
  ;
#pragma omp parallel reduction(inscan, + : a) // expected-error {{'inscan' modifier can be used only in 'omp for', 'omp simd', 'omp for simd', 'omp parallel for', or 'omp parallel for simd' directive}}
  ;
#pragma omp parallel reduction(+ : incomplete, ([10])p) // expected-error {{a reduction list item with incomplete type 'int[]'}} expected-error {{expected variable name, array element or array section}}
  ;
}

// complete to suppress an additional warning, but it's too late for pragmas
int incomplete[3];
