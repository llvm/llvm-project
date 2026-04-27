// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 -verify=expected,omp51 -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -verify=expected,omp60 -fsyntax-only %s

// Tests that the 'replayable' clause is accepted in OpenMP 6.0 and rejected in
// prior versions on all seven supported directives. Also tests duplicate-clause
// and invalid-condition diagnostics.

void foo() {}

void replayable_messages() {
  int A = 1;

  #pragma omp task replayable // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp task'}}
  {}

  #pragma omp taskloop replayable // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp taskloop'}}
  for (int i = 0; i < 10; ++i)
    {}

  #pragma omp taskwait replayable // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp taskwait'}}

  #pragma omp target replayable // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp target'}}
  {}

  #pragma omp target enter data map(to: A) replayable // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp target enter data'}}

  #pragma omp target exit data map(from: A) replayable // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp target exit data'}}

  #pragma omp target update to(A) replayable // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp target update'}}

  #pragma omp task replayable replayable // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp task'}} omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp task'}} expected-error {{directive '#pragma omp task' cannot contain more than one 'replayable' clause}}
  {}

  #pragma omp task replayable(foo()) // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp task'}} omp60-error {{value of type 'void' is not contextually convertible to 'bool'}}
  {}

  #pragma omp taskloop replayable(A > 0) // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp taskloop'}}
  for (int i = 0; i < 10; ++i)
    {}

  #pragma omp taskwait replayable(A > 0) // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp taskwait'}}

  #pragma omp target replayable(A > 0) // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp target'}}
  {}

  #pragma omp target enter data map(to: A) replayable(A > 0) // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp target enter data'}}

  #pragma omp target exit data map(from: A) replayable(A > 0) // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp target exit data'}}

  #pragma omp target update to(A) replayable(A > 0) // omp51-error {{unexpected OpenMP clause 'replayable' in directive '#pragma omp target update'}}
}
