// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=60 -ferror-limit 100 %s

// GH140338
// expected-warning@+2 {{expected string in 'clause message' - ignoring}}
// expected-error@+1 {{ERROR}}
#pragma omp error message(L"")
// expected-warning@+2 {{expected string in 'clause message' - ignoring}}
// expected-error@+1 {{ERROR}}
#pragma omp error message(L"bar")
// expected-warning@+2 {{expected string in 'clause message' - ignoring}}
// expected-warning@+1 {{WARNING}}
#pragma omp error severity(warning) message(L"bar")
// expected-warning@+2 {{expected string in 'clause message' - ignoring}}
// expected-error@+1 {{ERROR}}
#pragma omp error message(1)
// expected-warning@+2 {{expected string in 'clause message' - ignoring}}
// expected-error@+1 {{ERROR}}
#pragma omp error message(1.2)
#pragma omp error message("foo") // expected-error {{foo}}
#pragma omp error message(u8"foo") // expected-error {{foo}}

int foo(int i, const char *msg) {
// expected-warning@+2 {{expected string literal in 'clause message' - ignoring}}
// expected-error@+1 {{ERROR}}
#pragma omp error message(msg)
// expected-warning@+1 {{expected string in 'clause message' - ignoring}}
#pragma omp error at(execution) message(L"bar") // no error
  return i;
}
