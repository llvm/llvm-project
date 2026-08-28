// Split is not valid nested inside `teams` (host diagnostic).
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x c++ -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s

void g(void) {
#pragma omp teams
  {
// expected-error@+1 {{region cannot be closely nested inside 'teams' region}}
#pragma omp split counts(2, omp_fill)
    for (int i = 0; i < 10; ++i) {
    }
  }
}
