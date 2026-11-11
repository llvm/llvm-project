// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -x c++ -triple x86_64-unknown-unknown -fsyntax-only -verify %s

void braced_init(int n) {
#pragma omp for
  for (int i = {0}; i < n; i += {1})
    ;

#pragma omp for
  for (int i = {}; i < n; i -= {-1})
    ;

  // expected-error@+2 {{increment expression must cause 'i' to increase on each iteration of OpenMP for loop}} expected-note@+2 {{loop step is expected to be positive due to this condition}}
#pragma omp for
  for (int i = 0; i < n; i += {})
    ;
}
