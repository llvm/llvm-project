// RUN: %clang_cc1 -verify -fopenmp -std=c++11 -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++11 -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify=omp50 -fopenmp -fopenmp-version=50 -DOMP50 -std=c++11 -ferror-limit 100 -o - %s -Wuninitialized

#ifndef OMP50
void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}}

template <typename T, int C> // expected-note {{declared here}}
T tmain(T argc) {
  char **a;
  T z;
#pragma omp target
#pragma omp teams num_teams(C)
  foo();
#pragma omp target
#pragma omp teams num_teams(T) // expected-error {{'T' does not refer to a value}}
  foo();
#pragma omp target
#pragma omp teams num_teams // expected-error {{expected '(' after 'num_teams'}}
  foo();
#pragma omp target
#pragma omp teams num_teams( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp target
#pragma omp teams num_teams() // expected-error {{expected expression}}
  foo();
#pragma omp target
#pragma omp teams num_teams(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp target
#pragma omp teams num_teams(argc)) // expected-warning {{extra tokens at the end of '#pragma omp teams' are ignored}}
  foo();
#pragma omp target
#pragma omp teams num_teams(argc > 0 ? a[1] : a[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
#pragma omp target
#pragma omp teams num_teams(argc + argc+z)
  foo();
#pragma omp target
#pragma omp teams num_teams(argc), num_teams (argc+1) // expected-error {{directive '#pragma omp teams' cannot contain more than one 'num_teams' clause}}
  foo();
#pragma omp target
#pragma omp teams num_teams(S1) // expected-error {{'S1' does not refer to a value}}
  foo();
#pragma omp target
#pragma omp teams num_teams(-2) // expected-error {{argument to 'num_teams' clause must be a strictly positive integer value}}
  foo();
#pragma omp target
#pragma omp teams num_teams(-10u)
  foo();
#pragma omp target
#pragma omp teams num_teams(3.14) // expected-error 2 {{expression must have integral or unscoped enumeration type, not 'double'}}
  foo();
#pragma omp target
#pragma omp teams num_teams (1, 2, 3) // expected-error {{unexpected number of expressions in 'num_teams' clause (expected 1, have 3)}}
  foo();
#pragma omp target
#pragma omp teams thread_limit(1, 2, 3) // expected-error {{unexpected number of expressions in 'thread_limit' clause (expected 1, have 3)}}
  foo();

  return 0;
}

int main(int argc, char **argv) {
  int z;
#pragma omp target
#pragma omp teams num_teams // expected-error {{expected '(' after 'num_teams'}}
  foo();

#pragma omp target
#pragma omp teams num_teams ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();

#pragma omp target
#pragma omp teams num_teams () // expected-error {{expected expression}}
  foo();

#pragma omp target
#pragma omp teams num_teams (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();

#pragma omp target
#pragma omp teams num_teams (argc)) // expected-warning {{extra tokens at the end of '#pragma omp teams' are ignored}}
  foo();

#pragma omp target
#pragma omp teams num_teams (argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();

#pragma omp target
#pragma omp teams num_teams (argc + argc-z)
  foo();

#pragma omp target
#pragma omp teams num_teams (argc), num_teams (argc+1) // expected-error {{directive '#pragma omp teams' cannot contain more than one 'num_teams' clause}}
  foo();

#pragma omp target
#pragma omp teams num_teams (S1) // expected-error {{'S1' does not refer to a value}}
  foo();

#pragma omp target
#pragma omp teams num_teams (-2) // expected-error {{argument to 'num_teams' clause must be a strictly positive integer value}}
  foo();

#pragma omp target
#pragma omp teams num_teams (-10u)
  foo();

#pragma omp target
#pragma omp teams num_teams (3.14) // expected-error {{expression must have integral or unscoped enumeration type, not 'double'}}
  foo();

#pragma omp target
#pragma omp teams num_teams (1, 2, 3) // expected-error {{unexpected number of expressions in 'num_teams' clause (expected 1, have 3)}}
  foo();

#pragma omp target
#pragma omp teams thread_limit(1, 2, 3) // expected-error {{unexpected number of expressions in 'thread_limit' clause (expected 1, have 3)}}
  foo();

  return tmain<int, 10>(argc); // expected-note {{in instantiation of function template specialization 'tmain<int, 10>' requested here}}
}

// Test invalid syntax cases for num_teams lower-bound:upper-bound
void test_invalid_syntax() {
  int a = 1, b = 2, c = 3;

  // expected-error@+1 {{unexpected number of expressions in 'num_teams' clause (expected 1, have 3)}}
  #pragma omp teams num_teams(a, b, c)
  { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp teams num_teams(10:5)
  { }

  // expected-error@+1 {{unexpected number of expressions in 'num_teams' clause (expected 1, have 3)}}
  #pragma omp target teams num_teams(a, b, c)
  { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp target teams num_teams(8:3)
  { }

  // expected-error@+1 {{unexpected number of expressions in 'num_teams' clause (expected 1, have 3)}}
  #pragma omp target teams distribute num_teams(a, b, c)
  for (int i = 0; i < 100; ++i) { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp target teams distribute num_teams(15:7)
  for (int i = 0; i < 100; ++i) { }

  // expected-error@+1 {{unexpected number of expressions in 'num_teams' clause (expected 1, have 3)}}
  #pragma omp target teams distribute parallel for num_teams(a, b, c)
  for (int i = 0; i < 100; ++i) { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp target teams distribute parallel for num_teams(12:4)
  for (int i = 0; i < 100; ++i) { }

  // Test target teams distribute parallel for simd directive
  // expected-error@+1 {{unexpected number of expressions in 'num_teams' clause (expected 1, have 3)}}
  #pragma omp target teams distribute parallel for simd num_teams(a, b, c)
  for (int i = 0; i < 100; ++i) { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp target teams distribute parallel for simd num_teams(20:6)
  for (int i = 0; i < 100; ++i) { }

  // expected-error@+1 {{unexpected number of expressions in 'num_teams' clause (expected 1, have 3)}}
  #pragma omp target teams distribute simd num_teams(a, b, c)
  for (int i = 0; i < 100; ++i) { }
  // expected-error@+1 {{lower bound is greater than upper bound in 'num_teams' clause}}
  #pragma omp target teams distribute simd num_teams(9:2)
  for (int i = 0; i < 100; ++i) { }
}

// Test non-matching parentheses and brackets
void test_non_matching_delimiters() {
  int arr[10];
  int x = 5;

  // expected-error@+6 {{expected ')'}}
  // expected-error@+5 {{expected ')'}}
  // expected-error@+4 {{expected ')'}}
  // expected-note@+3 {{to match this '('}}
  // expected-note@+2 {{to match this '('}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp teams num_teams((x + 1:10)
  { }

  // expected-error@+2 {{expected ']'}}
  // expected-note@+1 {{to match this '['}}
  #pragma omp teams num_teams(arr[0:10)
  { }

  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp teams num_teams(x:((10 + 1))
  { }
}

// Test multi-level non-matching parentheses and brackets
void test_multi_level_non_matching_delimiters() {
  int arr[10][10];
  int x = 5, y = 10;

  // expected-error@+6 {{expected ')'}}
  // expected-error@+5 {{expected ')'}}
  // expected-error@+4 {{expected ')'}}
  // expected-note@+3 {{to match this '('}}
  // expected-note@+2 {{to match this '('}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp teams num_teams(((x + 1) * 2:10)
  { }

  // expected-error@+6 {{expected ')'}}
  // expected-error@+5 {{expected ')'}}
  // expected-error@+4 {{expected ')'}}
  // expected-note@+3 {{to match this '('}}
  // expected-note@+2 {{to match this '('}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp teams num_teams((x + (y - 1):10)
  { }
 
  // expected-error@+2 {{expected ']'}}
  // expected-note@+1 {{to match this '['}}
  #pragma omp teams num_teams((arr[0 + 1):10)
  { }

  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp teams num_teams(x:((y + 1) * 2)
  { }

  // expected-error@+4 {{expected ']'}}
  // expected-note@+3 {{to match this '['}}
  // expected-error@+2 {{expected ']'}}
  // expected-note@+1 {{to match this '['}}
  #pragma omp teams num_teams(arr[0][1:arr[2][3)
  { }
}
#endif

template<int Lower, int Upper>
void test_template_type_constants() {
  // expected-error@+2 {{lower bound is greater than upper bound in 'num_teams' clause}}
  // omp50-error@+1 {{'lower_bound' modifier in 'num_teams' clause requires OpenMP 5.1 or later}}
  #pragma omp teams num_teams(Lower:Upper)
  {}

  // omp50-error@+1 {{'lower_bound' modifier in 'num_teams' clause requires OpenMP 5.1 or later}}
  #pragma omp teams num_teams(Upper:Lower)
  {}
}

void instantiate_template_invalid() {
  test_template_type_constants<10, 5>(); // expected-note {{in instantiation of function template specialization 'test_template_type_constants<10, 5>' requested here}}
}


