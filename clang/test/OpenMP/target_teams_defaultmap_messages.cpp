// RUN: %clang_cc1 -verify -fopenmp %s

void foo() {
}

template <class T, typename S, int N, int ST>
T tmain(T argc, S **argv) {
  #pragma omp target teams defaultmap // expected-error {{expected '(' after 'defaultmap'}}
  foo();
  #pragma omp target teams defaultmap ( // expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams defaultmap () // expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target teams defaultmap (tofrom // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target teams defaultmap (tofrom: // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target teams defaultmap (tofrom) // expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target teams defaultmap (tofrom scalar) // expected-warning {{missing ':' after defaultmap modifier - ignoring}}
  foo();
  #pragma omp target teams defaultmap (tofrom, // expected-error {{expected ')'}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams defaultmap (scalar: // expected-error {{expected ')'}} expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams defaultmap (tofrom, scalar // expected-error {{expected ')'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  foo();

  return argc;
}

int main(int argc, char **argv) {
  #pragma omp target teams defaultmap // expected-error {{expected '(' after 'defaultmap'}}
  foo();
  #pragma omp target teams defaultmap ( // expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams defaultmap () // expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target teams defaultmap (tofrom // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target teams defaultmap (tofrom: // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target teams defaultmap (tofrom) // expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}}
  foo();
  #pragma omp target teams defaultmap (tofrom scalar) // expected-warning {{missing ':' after defaultmap modifier - ignoring}}
  foo();
  #pragma omp target teams defaultmap (tofrom, // expected-error {{expected ')'}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams defaultmap (scalar: // expected-error {{expected ')'}} expected-error {{expected 'tofrom' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target teams defaultmap (tofrom, scalar // expected-error {{expected ')'}} expected-warning {{missing ':' after defaultmap modifier - ignoring}} expected-error {{expected 'scalar' in OpenMP clause 'defaultmap'}} expected-note {{to match this '('}}
  foo();

  return tmain<int, char, 1, 0>(argc, argv);
}

