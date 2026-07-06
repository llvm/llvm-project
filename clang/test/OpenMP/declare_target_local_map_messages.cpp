// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ \
// RUN:   -triple x86_64-unknown-unknown %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ \
// RUN:   -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa %s

// OpenMP 6.0 [7.9.6]: device-local variable must not appear in map clause.

int x_local;
#pragma omp declare target local(x_local)

int arr_local[10];
#pragma omp declare target local(arr_local)

struct S {
  int a;
  int b;
};

S s_local;
#pragma omp declare target local(s_local)

void test_target() {
  // expected-error@+1 {{device-local variable 'x_local' is not allowed in 'map' clause}}
  #pragma omp target map(tofrom: x_local)
  { x_local = 1; }

  // expected-error@+1 {{device-local variable 'arr_local' is not allowed in 'map' clause}}
  #pragma omp target map(to: arr_local)
  { arr_local[0] = 2; }

  // expected-error@+1 {{device-local variable 'x_local' is not allowed in 'map' clause}}
  #pragma omp target map(from: x_local)
  { x_local = 3; }

  // expected-error@+1 {{device-local variable 'x_local' is not allowed in 'map' clause}}
  #pragma omp target map(alloc: x_local)
  { x_local = 4; }

  // Two device-local variables in one map clause.
  // expected-error@+2 {{device-local variable 'x_local' is not allowed in 'map' clause}}
  // expected-error@+1 {{device-local variable 'arr_local' is not allowed in 'map' clause}}
  #pragma omp target map(tofrom: x_local, arr_local)
  { x_local = arr_local[0]; }
}

void test_target_data() {
  // expected-error@+1 {{device-local variable 'x_local' is not allowed in 'map' clause}}
  #pragma omp target data map(tofrom: x_local)
  { }

  // expected-error@+1 {{device-local variable 'arr_local' is not allowed in 'map' clause}}
  #pragma omp target data map(alloc: arr_local)
  { }
}

void test_target_enter_data() {
  // expected-error@+1 {{device-local variable 'x_local' is not allowed in 'map' clause}}
  #pragma omp target enter data map(to: x_local)

  // expected-error@+1 {{device-local variable 'arr_local' is not allowed in 'map' clause}}
  #pragma omp target enter data map(alloc: arr_local)
}

void test_target_exit_data() {
  // expected-error@+1 {{device-local variable 'x_local' is not allowed in 'map' clause}}
  #pragma omp target exit data map(from: x_local)

  // expected-error@+1 {{device-local variable 'arr_local' is not allowed in 'map' clause}}
  #pragma omp target exit data map(delete: arr_local)
}
