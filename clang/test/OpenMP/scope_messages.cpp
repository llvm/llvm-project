// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 %s -verify=expected,omp51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 %s -verify=expected,omp51

void test1()
{
  int var1;
  int var2;
  int var3 = 1;

  // expected-error@+1 {{directive '#pragma omp scope' cannot contain more than one 'nowait' clause}} //omp51-error@+1{{unexpected OpenMP clause 'firstprivate' in directive '#pragma omp scope'}}
  #pragma omp scope private(var1) firstprivate(var3) nowait nowait
  { var1 = 123; ++var2; var3 = 2;}
}
