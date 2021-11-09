// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 %s -verify

int foobar() {
  return 1;
}

int main(int argc, char *argv[]) {
  // expected-note@+1 {{declared here}}
  int a;
  // expected-note@+1 {{declared here}}
  int b;
  // expected-note@+1 {{declared here}}
  int c;
  double f;
  int foo2[10];

// expected-error@+1 {{expected '(' after 'align'}}
#pragma omp allocate(a) align
// expected-error@+3 {{expected expression}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp allocate(a) align(
// expected-error@+1 {{expected expression}}
#pragma omp allocate(a) align()
// expected-error@+4 {{expected ')'}}
// expected-note@+3 {{to match this '('}}
// expected-error@+2 {{expression is not an integral constant expression}}
// expected-note@+1 {{read of non-const variable 'a' is not allowed in a constant expression}}
#pragma omp allocate(a) align(a
// expected-error@+2 {{expression is not an integral constant expression}}
// expected-note@+1 {{read of non-const variable 'b' is not allowed in a constant expression}}
#pragma omp allocate(a) align(b)
// expected-error@+2 {{expression is not an integral constant expression}}
// expected-note@+1 {{read of non-const variable 'c' is not allowed in a constant expression}}
#pragma omp allocate(a) align(c + 1)
// expected-error@+1 {{expected an OpenMP directive}}
#pragma omp align(2) allocate(a)
// expected-error@+1 {{directive '#pragma omp allocate' cannot contain more than one 'align' clause}}
#pragma omp allocate(a) align(2) align(4)
// expected-warning@+1 {{aligned clause will be ignored because the requested alignment is not a power of 2}}
#pragma omp allocate(a) align(9)
// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'double'}}
#pragma omp allocate(a) align(f)
}

// Verify appropriate errors when using templates.
template <typename T, unsigned size, unsigned align>
T run() {
  T foo[size];
// expected-warning@+1 {{aligned clause will be ignored because the requested alignment is not a power of 2}}
#pragma omp allocate(foo) align(align)
  return foo[0];
}

int template_test() {
  double d;
  // expected-note@+1 {{in instantiation of function template specialization 'run<double, 10U, 3U>' requested here}}
  d = run<double, 10, 3>();
  return 0;
}
