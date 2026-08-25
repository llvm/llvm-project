// Test for ompx_name clause error checking
// RUN: %clang_cc1 -std=c++20 -verify -fopenmp %s

static void foo() {
}

void bar() {
  int x = 5;

  // expected-error@+1 {{argument to 'ompx_name' clause must be a string literal}}
  #pragma omp target ompx_name(x)
  {
  }

  // expected-error@+1 {{argument to 'ompx_name' clause must be a string literal}}
  #pragma omp target ompx_name(123)
  {
  }

  // This should work - string literal
  #pragma omp target ompx_name("valid_name")
  {
  }

// expected-note@+1 {{previous use of this kernel name is here}}
#pragma omp target ompx_name("baz")
  foo();

// expected-error@+1 {{OpenMP target kernel name 'baz' is used more than once in this translation unit}}
#pragma omp target ompx_name("baz")
  foo();

#pragma omp target ompx_name(foo) // expected-error {{argument to 'ompx_name' clause must be a string literal}}
  foo();

#pragma omp target ompx_name("foo", "bar") // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
}

consteval const char* getStr() {
  return "foobar3"; 
}

void foobar() {
// CHECK: define {{.*}} @foobar3(
  #pragma omp target ompx_name(getStr())  // expected-error {{argument to 'ompx_name' clause must be a string literal}}
  {}
}

template<typename T>
void TTT() {
// expected-note@+2 {{previous use of this kernel name is here}}
// expected-error@+1 {{OpenMP target kernel name 'template' is used more than once in this translation unit}}
  #pragma omp target ompx_name("template")
  {}
}

void test2() {
// expected-note@+1 {{in instantiation of function template specialization 'TTT<int>' requested here}}
  TTT<int>();
  TTT<float>();
}
