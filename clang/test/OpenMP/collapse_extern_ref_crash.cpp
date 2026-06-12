// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 %s

// This test verify two behaviors:
// 1. No crash when the loop bound is an extern reference (fixed null pointer
// dereference).
// 2. Proper diagnostic when a nested loop reuses the outer loop's induction
// variable


extern int &dim;
auto test1() {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < dim; ++i) {
    // expected-error@+1{{loop iteration variable 'i' cannot be reused in a nested loop of a collapsed loop nest}}
    for (i = 0; i < 10; i++) {
      int dummy;
    }
  }
}

auto test2() {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < dim; ++i) {
    for (int i = 0; i < 10; i++) {
      int dummy;
    }
  }
}

int dim_storage = 10;
int &dim = dim_storage;
auto test3() {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < 10; j++) {
      int dummy;
    }
  }
}

struct Iterator {
  int value;
  Iterator& operator=(int v) { value = v; return *this; }
  bool operator<(int n) const { return value < n; }
  Iterator& operator++() { ++value; return *this; }
};

Iterator i;
auto test4() {
#pragma omp parallel for collapse(2)
  for (i = 0; i < dim; ++i) {
    // expected-error@+1{{loop iteration variable 'i' cannot be reused in a nested loop of a collapsed loop nest}}
    for (i = 0; i < 10; ++i) {
      int dummy;
    }
  }
}

auto test5() {
#pragma omp parallel for collapse(2)
  for (i = 0; i < dim; ++i) {
    // expected-error@+1{{loop iteration variable 'i' cannot be reused in a nested loop of a collapsed loop nest}}
    for (i = 0; i < 10; ++i) {
      int dummy;
    }
  }
}
