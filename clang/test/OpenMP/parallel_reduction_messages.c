// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ferror-limit 150 -o - %s

int incomplete[];

void test() {
  int a;
#pragma omp parallel reduction( // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-warning {{missing ':' after reduction identifier - ignoring}} expected-note {{to match this '('}}
  ;
#pragma omp parallel reduction(unknown // expected-error {{expected expression}} expected-error {{expected ')'}} expected-warning {{missing ':' after reduction identifier - ignoring}} expected-note {{to match this '('}}
  ;
#pragma omp parallel reduction(default, // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-warning {{missing ':' after reduction identifier - ignoring}} expected-note {{to match this '('}}
  ;
#pragma omp parallel reduction(unknown, +: a) // expected-error {{expected 'default' in OpenMP clause 'reduction'}}
  ;
#pragma omp parallel reduction(default, + : a)
  ;
#pragma omp parallel reduction(+ : incomplete) // expected-error {{a reduction list item with incomplete type 'int []'}}
  ;
}

// complete to suppress an additional warning, but it's too late for pragmas
int incomplete[3];
