// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 %s

// This test verifies two behaviors:
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

struct S {
  int x;
  void test6() {
#pragma omp parallel for collapse(2)
    for (this->x = 0; this->x < 10; ++this->x) {
      // expected-error@+1{{loop iteration variable 'x' cannot be reused in a nested loop of a collapsed loop nest}}
      for (this->x = 0; this->x < 10; ++this->x) {
	int dummy;
      }
    }
  }
};

struct S2 {
  int x;
  int y;
  void test7() {
#pragma omp parallel for collapse(2)
    for (this->x = 0; this->x < 10; ++this->x) {
      for (this->y = 0; this->y < 10; ++this->y) {
	int dummy;
      }
    }
  }
};

struct S3 {
  Iterator x;
  void test8() {
#pragma omp parallel for collapse(2)
    for (this->x = 0; this->x < 10; ++this->x) {
      // expected-error@+1{{loop iteration variable 'x' cannot be reused in a nested loop of a collapsed loop nest}}
      for (this->x = 0; this->x < 10; ++this->x) {
	int dummy;
      }
    }
  }
};

struct S4 {
  int x;
  int &r;
  S4() : x(0), r(x) {}

  void test_ref() {
#pragma omp parallel for collapse(2)
    for (this->r = 0; this->r < 10; ++this->r) {
      // expected-error@+1{{loop iteration variable 'r' cannot be reused in a nested loop of a collapsed loop nest}}
      for (this->r = 0; this->r < 10; ++this->r) {
	int dummy;
      }
    }
  }
};

auto test9(int *out) {
  int storage = 0;
  int &r = storage;

#pragma omp parallel for collapse(2)
  for (r = 0; r < 10; ++r) {
    // expected-error@+1{{loop iteration variable 'r' cannot be reused in a nested loop of a collapsed loop nest}}
    for (r = 0; r < 10; ++r) {
      out[r] = r;
    }
  }
}
