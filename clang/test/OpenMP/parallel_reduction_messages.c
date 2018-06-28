// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 150 -o - %s

int incomplete[];

void test() {
#pragma omp parallel reduction(+ : incomplete) // expected-error {{a reduction list item with incomplete type 'int []'}}
  ;
}

// complete to suppress an additional warning, but it's too late for pragmas
int incomplete[3];
