// RUN: %clang -fopenmp -O -g -x c %s -S -disable-output -o %t

// Do not crash ;)

void foo()
{
#pragma omp critical
  ;
}

void bar()
{
  foo();
  foo();
}
