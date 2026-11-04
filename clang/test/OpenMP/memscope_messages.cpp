// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp -fopenmp-version=51 -ferror-limit 100 %s
// RUN: %clang_cc1 -verify=expected,omp60 -fopenmp -fopenmp-version=60 -ferror-limit 100 %s

// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp-simd -fopenmp-version=51 -ferror-limit 100 %s
// RUN: %clang_cc1 -verify=expected,omp60 -fopenmp-simd -fopenmp-version=60 -ferror-limit 100 %s

int x, v;

void foo() {
#pragma omp flush memscope(device) // omp51-error {{unexpected OpenMP clause 'memscope' in directive '#pragma omp flush'}}
#pragma omp flush release memscope(all) // omp51-error {{unexpected OpenMP clause 'memscope' in directive '#pragma omp flush'}}
#pragma omp flush memscope(cgroup) acquire // omp51-error {{unexpected OpenMP clause 'memscope' in directive '#pragma omp flush'}}
#pragma omp flush memscope(system) // omp51-error {{unexpected OpenMP clause 'memscope' in directive '#pragma omp flush'}} omp60-error {{expected 'all', 'cgroup' or 'device' in OpenMP clause 'memscope'}}
#pragma omp flush memscope( // omp51-error {{unexpected OpenMP clause 'memscope' in directive '#pragma omp flush'}} omp60-error {{expected 'all', 'cgroup' or 'device' in OpenMP clause 'memscope'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp flush memscope // omp51-error {{unexpected OpenMP clause 'memscope' in directive '#pragma omp flush'}} expected-error {{expected '(' after 'memscope'}}
#pragma omp flush memscope(all) memscope(device) // omp51-error 2 {{unexpected OpenMP clause 'memscope' in directive '#pragma omp flush'}} expected-error {{directive '#pragma omp flush' cannot contain more than one 'memscope' clause}}
#pragma omp flush(x) memscope(all) // omp51-error {{unexpected OpenMP clause 'memscope' in directive '#pragma omp flush'}}
#pragma omp flush(x) release memscope(all) // omp51-error {{unexpected OpenMP clause 'memscope' in directive '#pragma omp flush'}} expected-error {{'flush' directive with memory order clause 'release' cannot have the list}} expected-note {{memory order clause 'release' is specified here}}
#pragma omp atomic read memscope(device) // omp51-error {{unexpected OpenMP clause 'memscope' in directive '#pragma omp atomic'}}
  v = x;
#pragma omp atomic seq_cst memscope(all) // omp51-error {{unexpected OpenMP clause 'memscope' in directive '#pragma omp atomic'}}
  x += v;
#pragma omp atomic memscope(cgroup) memscope(all) // omp51-error 2 {{unexpected OpenMP clause 'memscope' in directive '#pragma omp atomic'}} expected-error {{directive '#pragma omp atomic' cannot contain more than one 'memscope' clause}}
  x++;
#pragma omp barrier memscope(all) // expected-error {{unexpected OpenMP clause 'memscope' in directive '#pragma omp barrier'}}
}
