// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++20 -fopenmp -fopenmp-version=61 -fsyntax-only -verify %s

void func() {
  // expected-error@+1 {{'depth' clause is not yet supported on '#pragma omp fuse' directive}}
  #pragma omp fuse depth(2)
  {
    for (int i = 0; i < 7; ++i)
      ;
    for (int j = 0; j < 9; ++j)
      ;
  }
}
