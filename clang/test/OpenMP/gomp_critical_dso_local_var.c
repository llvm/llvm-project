// RUN: %clang_cc1 -fopenmp -x c -emit-llvm %s -o - | FileCheck %s --check-prefix=DSO_LOCAL

// DSO_LOCAL-DAG: @.gomp_critical_user_.var = common dso_local global [8 x i32] zeroinitializer, align 8
int omp_critical_test()
{
  int sum;
  int known_sum;

  sum=0;
#pragma omp parallel
  {
    int mysum=0;
    int i;
#pragma omp for
    for (i = 0; i < 1000; i++)
      mysum = mysum + i;
#pragma omp critical
    sum = mysum +sum;
  }
  known_sum = 999 * 1000 / 2;
  return (known_sum == sum);
}
