// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// expected-no-diagnostics

void foo();

// CHECK-LABEL: void test_lower_upper_bound()
void test_lower_upper_bound() {
  int lower = 2, upper = 8;
  // CHECK: #pragma omp teams num_teams(2:8)
  #pragma omp teams num_teams(2:8)
  { foo(); }

  // CHECK: #pragma omp teams num_teams(lower:upper)
  #pragma omp teams num_teams(lower:upper)
  { foo(); }

  // CHECK: #pragma omp target teams num_teams(1:10)
  #pragma omp target teams num_teams(1:10)
  { foo(); }

  // CHECK: #pragma omp target teams distribute num_teams(3:6)
  #pragma omp target teams distribute num_teams(3:6)
  for (int i = 0; i < 100; ++i) { }
}

// CHECK-LABEL: void test_various_directives()
void test_various_directives() {
  int lb = 4, ub = 12;

  // CHECK: #pragma omp target teams distribute parallel for num_teams(lb:ub)
  #pragma omp target teams distribute parallel for num_teams(lb:ub)
  for (int i = 0; i < 100; ++i) { }

  // CHECK: #pragma omp target teams distribute parallel for simd num_teams(2:16)
  #pragma omp target teams distribute parallel for simd num_teams(2:16)
  for (int i = 0; i < 100; ++i) { }

  // CHECK: #pragma omp target teams distribute simd num_teams(1:8)
  #pragma omp target teams distribute simd num_teams(1:8)
  for (int i = 0; i < 100; ++i) { }
}

