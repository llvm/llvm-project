// RUN: %clang_cc1 -verify=expected -fopenmp -ferror-limit 100 %s -Wuninitialized

int mixed() {
  int x = 0;
  int d = 4;

#pragma omp nothing
  x=d;

  if(!x) {
#pragma omp nothing
    x=d;
  }

// expected-error@+2 {{#pragma omp nothing' cannot be an immediate substatement}}
  if(!x)
#pragma omp nothing
    x=d;

// expected-warning@+2 {{extra tokens at the end of '#pragma omp nothing' are ignored}}
  if(!x) {
#pragma omp nothing seq_cst
    x=d;
  }

  return 0;
}
