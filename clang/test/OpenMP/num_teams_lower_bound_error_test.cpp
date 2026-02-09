// RUN: %clang_cc1 -verify -fopenmp -std=c++11 %s -Wuninitialized

// Test invalid syntax cases for num_teams lower-bound:upper-bound
void test_invalid_syntax() {
  int a = 1, b = 2, c = 3;

  // expected-error@+1 {{only two expression allowed in 'num_teams' clause}}
  #pragma omp teams num_teams(a, b, c)
  { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp teams num_teams(10:5)
  { }

  // expected-error@+1 {{only two expression allowed in 'num_teams' clause}}
  #pragma omp target teams num_teams(a, b, c)
  { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp target teams num_teams(8:3)
  { }

  // expected-error@+1 {{only two expression allowed in 'num_teams' clause}}
  #pragma omp target teams distribute num_teams(a, b, c)
  for (int i = 0; i < 100; ++i) { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp target teams distribute num_teams(15:7)
  for (int i = 0; i < 100; ++i) { }

  // expected-error@+1 {{only two expression allowed in 'num_teams' clause}}
  #pragma omp target teams distribute parallel for num_teams(a, b, c)
  for (int i = 0; i < 100; ++i) { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp target teams distribute parallel for num_teams(12:4)
  for (int i = 0; i < 100; ++i) { }

  // Test target teams distribute parallel for simd directive
  // expected-error@+1 {{only two expression allowed in 'num_teams' clause}}
  #pragma omp target teams distribute parallel for simd num_teams(a, b, c)
  for (int i = 0; i < 100; ++i) { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp target teams distribute parallel for simd num_teams(20:6)
  for (int i = 0; i < 100; ++i) { }

  // expected-error@+1 {{only two expression allowed in 'num_teams' clause}}
  #pragma omp target teams distribute simd num_teams(a, b, c)
  for (int i = 0; i < 100; ++i) { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp target teams distribute simd num_teams(9:2)
  for (int i = 0; i < 100; ++i) { }
}
