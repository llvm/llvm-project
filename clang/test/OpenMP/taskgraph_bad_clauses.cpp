// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp -fopenmp-version=51 -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -verify=expected,omp60 -fopenmp -fopenmp-version=60 -ferror-limit 100 -o - %s

int main() {
  int data[10];
#pragma omp taskgraph map(tofrom: data[0:10]) // expected-error {{unexpected OpenMP clause 'map' in directive '#pragma omp taskgraph'}} omp51-error {{unexpected OpenMP directive '#pragma omp taskgraph'}}
  {
  }
#pragma omp taskgraph depend(inout: data) // expected-error {{unexpected OpenMP clause 'depend' in directive '#pragma omp taskgraph'}} omp51-error {{unexpected OpenMP directive '#pragma omp taskgraph'}}
  {
  }
#pragma omp taskgraph if(taskgraph: 1) // omp51-error {{unexpected OpenMP directive '#pragma omp taskgraph'}}
  {
  }
#pragma omp taskgraph if(cancel: 1) // omp51-error {{unexpected OpenMP directive '#pragma omp taskgraph'}} expected-error {{directive name modifier 'cancel' is not allowed for '#pragma omp taskgraph'}}
  {
  }
  return 0;
}
