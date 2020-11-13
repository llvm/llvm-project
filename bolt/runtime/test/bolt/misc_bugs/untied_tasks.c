// RUN: %libomp-compile-and-run
// REQUIRES: abt
#include "omp_testsuite.h"

int test_omp_untied_tasks()
{
  // https://github.com/pmodels/bolt/issues/49
  int val = 0;
  #pragma omp parallel
  #pragma omp master
  {
    #pragma omp task untied
    { val = 1; }
  }
  return val;
}

int test_omp_tied_tasks()
{
  int val = 0;
  #pragma omp parallel
  #pragma omp master
  {
    #pragma omp task
    { val = 1; }
  }
  return val;
}

int test_omp_tied_and_untied_tasks()
{
  int val1 = 0;
  int val2 = 0;
  #pragma omp parallel
  #pragma omp master
  {
    #pragma omp task
    { val1 = 1; }
    #pragma omp task untied
    { val2 = 1; }
  }
  return val1 == 1 && val2 == 1;
}

int main()
{
  int i;
  int num_failed = 0;
  for (i = 0; i < REPETITIONS; i++) {
    if (!test_omp_untied_tasks()) {
      num_failed++;
    }
    if (!test_omp_tied_tasks()) {
      num_failed++;
    }
    if (!test_omp_tied_and_untied_tasks()) {
      num_failed++;
    }
  }
  return num_failed;
}
