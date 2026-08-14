// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -verify=expected -fsyntax-only %s

int main() {
#pragma omp taskgraph
  {
#pragma omp taskwait // expected-error {{directive '#pragma omp taskwait' within '#pragma omp taskgraph' must use 'depend' clause to be task-generating}}
  }

#pragma omp taskwait

  return 0;
}
