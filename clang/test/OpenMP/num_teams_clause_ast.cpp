// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

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

// CHECK-LABEL: void test_nested_expressions()
void test_nested_expressions() {
  int arr[10][10];
  int x = 5, y = 10;

  // CHECK: #pragma omp teams num_teams(arr[0][0]:arr[1][1])
  #pragma omp teams num_teams(arr[0][0]:arr[1][1])
  { foo(); }

  // CHECK: #pragma omp teams num_teams(arr[x][0]:arr[y][1])
  #pragma omp teams num_teams(arr[x][0]:arr[y][1])
  { foo(); }

  // CHECK: #pragma omp teams num_teams((x + 1):(y - 1))
  #pragma omp teams num_teams((x + 1):(y - 1))
  { foo(); }

}

// CHECK-LABEL: void test_multi_level_matching_delimiters()
void test_multi_level_matching_delimiters() {
  int arr[10][10];
  int x = 5, y = 10;
  
  // CHECK: #pragma omp teams num_teams(((x + 1) * 2):((y - 1) * 3))
  #pragma omp teams num_teams(((x + 1) * 2):((y - 1) * 3))
  { foo(); }


  // CHECK: #pragma omp teams num_teams(arr[arr[0][0]][0]:arr[arr[1][1]][1])
  #pragma omp teams num_teams(arr[arr[0][0]][0]:arr[arr[1][1]][1])
  { foo(); }

  // CHECK: #pragma omp teams num_teams((arr[x][y] + 1):(arr[y][x] - 1))
  #pragma omp teams num_teams((arr[x][y] + 1):(arr[y][x] - 1))
  { foo(); }

  // CHECK: #pragma omp teams num_teams((x + (y * 2)):((x * 2) + y))
  #pragma omp teams num_teams((x + (y * 2)):((x * 2) + y))
  { foo(); }

  // CHECK: #pragma omp teams num_teams((arr[0][1] + arr[2][3]):(arr[4][5] + arr[6][7]))
  #pragma omp teams num_teams((arr[0][1] + arr[2][3]):(arr[4][5] + arr[6][7]))
  { foo(); }
}

// Template tests
// CHECK-LABEL: template <typename T> void test_template_type(T lower, T upper)
template<typename T>
void test_template_type(T lower, T upper) {
  // CHECK: #pragma omp teams num_teams(lower:upper)
  #pragma omp teams num_teams(lower:upper)
  {}
}

#endif
