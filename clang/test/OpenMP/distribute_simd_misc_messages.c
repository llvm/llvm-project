// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-version=45 -verify=expected,omp45 %s -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-version=50 -verify=expected,omp50,omp-50-and-later,omp-50-and-later-temporal,omp50-temporal,omp50-and-later-var %s -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-version=60 -verify=expected,omp60,omp-clause,omp-50-and-later,omp-50-and-later-temporal,omp50-temporal,omp50-and-later-var %s -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -fopenmp -verify=expected,omp51,omp-clause,omp-50-and-later,omp-50-and-later-temporal,omp50-temporal,omp50-and-later-var %s -Wuninitialized

// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -fopenmp-version=45 -verify=expected,omp45 %s -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -fopenmp-version=50 -verify=expected,omp50,omp-50-and-later,omp-50-and-later-temporal,omp50-temporal,omp50-and-later-var %s -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -fopenmp-version=60 -verify=expected,omp60,omp-clause,omp-50-and-later,omp-50-and-later-temporal,omp50-temporal,omp50-and-later-var %s -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -verify=expected,omp51,omp-clause,omp-50-and-later,omp-50-and-later-temporal,omp50-temporal,omp50-and-later-var %s -Wuninitialized

void xxx(int argc) {
  // expected-note@+1 {{initialize the variable 'x' to silence this warning}}
  int x;
#pragma omp distribute simd
  for (int i = 0; i < 10; ++i)
    // expected-warning@+1 {{variable 'x' is uninitialized when used here}}
    argc = x;
}

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp distribute simd'}}
#pragma omp distribute simd

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp distribute simd'}}
#pragma omp distribute simd foo

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp distribute simd'}}
#pragma omp distribute simd safelen(4)

void test_no_clause(void) {
  int i;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-error@+2 {{statement after '#pragma omp distribute simd' must be a for loop}}
#pragma omp distribute simd
  ++i;
}

void test_branch_protected_scope(void) {
  int i = 0;
L1:
  ++i;

  int x[24];

#pragma omp target
#pragma omp teams
#pragma omp distribute simd
  for (i = 0; i < 16; ++i) {
    if (i == 5)
      // expected-error@+1 {{use of undeclared label 'L1'}}
      goto L1;
    else if (i == 6)
      // expected-error@+1 {{cannot return from OpenMP region}}
      return;
    else if (i == 7)
      goto L2;
    else if (i == 8) {
    L2:
      x[i]++;
    }
  }

  if (x[0] == 0)
    // expected-error@+1 {{use of undeclared label 'L2'}}
    goto L2;
  else if (x[1] == 1)
    goto L1;
}

void test_invalid_clause(void) {
  int i;
#pragma omp target
#pragma omp teams
// expected-warning@+1 {{extra tokens at the end of '#pragma omp distribute simd' are ignored}}
#pragma omp distribute simd foo bar
  for (i = 0; i < 16; ++i)
    ;
}

void test_non_identifiers(void) {
  int i, x;

#pragma omp target
#pragma omp teams
// expected-warning@+1 {{extra tokens at the end of '#pragma omp distribute simd' are ignored}}
#pragma omp distribute simd;
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-warning@+1 {{extra tokens at the end of '#pragma omp distribute simd' are ignored}}
#pragma omp distribute simd private(x);
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-warning@+1 {{extra tokens at the end of '#pragma omp distribute simd' are ignored}}
#pragma omp distribute simd, private(x);
  for (i = 0; i < 16; ++i)
    ;
}

extern int foo(void);
void test_safelen(void) {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected '('}}
#pragma omp distribute simd safelen
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd safelen(
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd safelen()
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd safelen(,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}  expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd safelen(, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-warning@+2 {{extra tokens at the end of '#pragma omp distribute simd' are ignored}}
// expected-error@+1 {{expected '('}}
#pragma omp distribute simd safelen 4)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd safelen(4
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd safelen(4,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd safelen(4, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// xxpected-error@+1 {{expected expression}}
#pragma omp distribute simd safelen(4)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd safelen(4 4)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd safelen(4, , 4)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd safelen(4)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd safelen(4, 8)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{integer constant expression}}
#pragma omp distribute simd safelen(2.5)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{integer constant expression}}
#pragma omp distribute simd safelen(foo())
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{argument to 'safelen' clause must be a strictly positive integer value}}
#pragma omp distribute simd safelen(-5)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{argument to 'safelen' clause must be a strictly positive integer value}}
#pragma omp distribute simd safelen(0)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{argument to 'safelen' clause must be a strictly positive integer value}}
#pragma omp distribute simd safelen(5 - 5)
  for (i = 0; i < 16; ++i)
    ;
}

void test_simdlen(void) {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected '('}}
#pragma omp distribute simd simdlen
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd simdlen(
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd simdlen()
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd simdlen(,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}  expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd simdlen(, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-warning@+2 {{extra tokens at the end of '#pragma omp distribute simd' are ignored}}
// expected-error@+1 {{expected '('}}
#pragma omp distribute simd simdlen 4)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd simdlen(4
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd simdlen(4,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd simdlen(4, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd simdlen(4)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd simdlen(4 4)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd simdlen(4, , 4)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd simdlen(4)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd simdlen(4, 8)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{integer constant expression}}
#pragma omp distribute simd simdlen(2.5)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{integer constant expression}}
#pragma omp distribute simd simdlen(foo())
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{argument to 'simdlen' clause must be a strictly positive integer value}}
#pragma omp distribute simd simdlen(-5)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{argument to 'simdlen' clause must be a strictly positive integer value}}
#pragma omp distribute simd simdlen(0)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{argument to 'simdlen' clause must be a strictly positive integer value}}
#pragma omp distribute simd simdlen(5 - 5)
  for (i = 0; i < 16; ++i)
    ;
}

void test_safelen_simdlen(void) {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{the value of 'simdlen' parameter must be less than or equal to the value of the 'safelen' parameter}}
#pragma omp distribute simd simdlen(6) safelen(5)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-error@+1 {{the value of 'simdlen' parameter must be less than or equal to the value of the 'safelen' parameter}}
#pragma omp distribute simd safelen(5) simdlen(6)
  for (i = 0; i < 16; ++i)
    ;
}

void test_collapse(void) {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected '('}}
#pragma omp distribute simd collapse
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd collapse(
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd collapse()
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd collapse(,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}  expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd collapse(, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-warning@+2 {{extra tokens at the end of '#pragma omp distribute simd' are ignored}}
// expected-error@+1 {{expected '('}}
#pragma omp distribute simd collapse 4)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4
  for (i = 0; i < 16; ++i)
// expected-error@+1 {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4,
  for (i = 0; i < 16; ++i)
// expected-error@+1 {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4, )
  for (i = 0; i < 16; ++i)
// expected-error@+1 {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
    ;
#pragma omp target
#pragma omp teams
// expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4)
  for (i = 0; i < 16; ++i)
// expected-error@+1 {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
    ;
#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected ')'}}
// expected-note@+2 {{to match this '('}}
// expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4 4)
  for (i = 0; i < 16; ++i)
// expected-error@+1 {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
    ;
#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected ')'}}
// expected-note@+2 {{to match this '('}}
// expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4, , 4)
  for (i = 0; i < 16; ++i)
// expected-error@+1 {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd collapse(4)
  for (int i1 = 0; i1 < 16; ++i1)
    for (int i2 = 0; i2 < 16; ++i2)
      for (int i3 = 0; i3 < 16; ++i3)
        for (int i4 = 0; i4 < 16; ++i4)
          foo();
#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected ')'}}
// expected-note@+2 {{to match this '('}}
// expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4, 8)
  for (i = 0; i < 16; ++i)
// expected-error@+1 {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{integer constant expression}}
#pragma omp distribute simd collapse(2.5)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{integer constant expression}}
#pragma omp distribute simd collapse(foo())
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{argument to 'collapse' clause must be a strictly positive integer value}}
#pragma omp distribute simd collapse(-5)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{argument to 'collapse' clause must be a strictly positive integer value}}
#pragma omp distribute simd collapse(0)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{argument to 'collapse' clause must be a strictly positive integer value}}
#pragma omp distribute simd collapse(5 - 5)
  for (i = 0; i < 16; ++i)
    ;
// expected-note@+3 2 {{defined as reduction}}
#pragma omp target
#pragma omp teams
#pragma omp distribute simd collapse(2) reduction(+ : i)
  for (i = 0; i < 16; ++i) // expected-error {{loop iteration variable in the associated loop of 'omp distribute simd' directive may not be reduction, predetermined as lastprivate}}
    // expected-note@+1 {{variable with automatic storage duration is predetermined as private; perhaps you forget to enclose 'omp for' directive into a parallel or another task region?}}
    for (int j = 0; j < 16; ++j)
// expected-error@+2 2 {{reduction variable must be shared}}
// expected-error@+1 {{OpenMP constructs may not be nested inside a simd region}}
#pragma omp for reduction(+ : i, j)
      for (int k = 0; k < 16; ++k)
        i += j;

#pragma omp target
#pragma omp teams
  for (i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
#pragma omp distribute simd reduction(+ : i, j)
      for (int k = 0; k < 16; ++k)
        i += j;
}

void test_linear(void) {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd linear(
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected expression}}
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd linear(,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected expression}}
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd linear(, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd linear()
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd linear(int)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected variable name}}
#pragma omp distribute simd linear(0)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{use of undeclared identifier 'x'}}
#pragma omp distribute simd linear(x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{use of undeclared identifier 'x'}}
// expected-error@+1 {{use of undeclared identifier 'y'}}
#pragma omp distribute simd linear(x, y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+3 {{use of undeclared identifier 'x'}}
// expected-error@+2 {{use of undeclared identifier 'y'}}
// expected-error@+1 {{use of undeclared identifier 'z'}}
#pragma omp distribute simd linear(x, y, z)
  for (i = 0; i < 16; ++i)
    ;
}

void test_aligned(void) {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd aligned(
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected expression}}
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd aligned(,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected expression}}
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd aligned(, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd aligned()
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd aligned(int)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected variable name}}
#pragma omp distribute simd aligned(0)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{use of undeclared identifier 'x'}}
#pragma omp distribute simd aligned(x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{use of undeclared identifier 'x'}}
// expected-error@+1 {{use of undeclared identifier 'y'}}
#pragma omp distribute simd aligned(x, y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+3 {{use of undeclared identifier 'x'}}
// expected-error@+2 {{use of undeclared identifier 'y'}}
// expected-error@+1 {{use of undeclared identifier 'z'}}
#pragma omp distribute simd aligned(x, y, z)
  for (i = 0; i < 16; ++i)
    ;

  int *x, y, z[25]; // expected-note 4 {{'y' defined here}}
#pragma omp target
#pragma omp teams
#pragma omp distribute simd aligned(x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd aligned(z)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd aligned(x :)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd aligned(x :, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd aligned(x : 1)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd aligned(x : 2 * 2)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd aligned(x : 1, y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd aligned(x : 1, y, z : 1)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-error@+1 {{argument of aligned clause should be array or pointer, not 'int'}}
#pragma omp distribute simd aligned(x, y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{argument of aligned clause should be array or pointer, not 'int'}}
#pragma omp distribute simd aligned(x, y, z)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-note@+2 {{defined as aligned}}
// expected-error@+1 {{a variable cannot appear in more than one aligned clause}}
#pragma omp distribute simd aligned(x) aligned(z, x)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-note@+3 {{defined as aligned}}
// expected-error@+2 {{a variable cannot appear in more than one aligned clause}}
// expected-error@+1 2 {{argument of aligned clause should be array or pointer, not 'int'}}
#pragma omp distribute simd aligned(x, y, z) aligned(y, z)
  for (i = 0; i < 16; ++i)
    ;
}

void test_private(void) {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected expression}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd private(
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected ')'}}
//  expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp distribute simd private(,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 2 {{expected expression}}
#pragma omp distribute simd private(, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd private()
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd private(int)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected variable name}}
#pragma omp distribute simd private(0)
  for (i = 0; i < 16; ++i)
    ;

  int x, y, z;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd private(x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd private(x, y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd private(x, y, z)
  for (i = 0; i < 16; ++i) {
    x = y * i + z;
  }
}

void test_firstprivate(void) {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected ')'}}
//  expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd firstprivate(
  for (i = 0; i < 16; ++i)
    ;
}

void test_lastprivate(void) {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected ')'}}
// expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd lastprivate(
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected ')'}}
// expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp distribute simd lastprivate(,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 2 {{expected expression}}
#pragma omp distribute simd lastprivate(, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd lastprivate()
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd lastprivate(int)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected variable name}}
#pragma omp distribute simd lastprivate(0)
  for (i = 0; i < 16; ++i)
    ;

  int x, y, z;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(x, y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd lastprivate(x, y, z)
  for (i = 0; i < 16; ++i)
    ;
}

void test_reduction(void) {
  int i, x, y;
#pragma omp target
#pragma omp teams
// expected-error@+4 {{expected ')'}}
// expected-note@+3 {{to match this '('}}
// expected-error@+2 {{expected identifier}}
// expected-warning@+1 {{missing ':' after reduction identifier - ignoring}}
#pragma omp distribute simd reduction(
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected identifier}}
// expected-warning@+1 {{missing ':' after reduction identifier - ignoring}}
#pragma omp distribute simd reduction()
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected expression}}
// expected-warning@+1 {{missing ':' after reduction identifier - ignoring}}
#pragma omp distribute simd reduction(x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected identifier}}
#pragma omp distribute simd reduction( : x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+4 {{expected ')'}}
// expected-note@+3 {{to match this '('}}
// expected-error@+2 {{expected identifier}}
// expected-warning@+1 {{missing ':' after reduction identifier - ignoring}}
#pragma omp distribute simd reduction(,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+4 {{expected ')'}}
// expected-note@+3 {{to match this '('}}
// expected-error@+2 {{expected expression}}
// expected-warning@+1 {{missing ':' after reduction identifier - ignoring}}
#pragma omp distribute simd reduction(+
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected ')'}}
//  expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd reduction(+:
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd reduction(+ :)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd reduction(+ :, y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd reduction(+ : x, + : y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected identifier}}
#pragma omp distribute simd reduction(% : x)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(+ : x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(* : x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
  // omp60-error@+1 {{incorrect reduction identifier, expected one of '+', '*', '&', '|', '^', '&&', '||', 'min' or 'max' or declare reduction for type 'int'}}
#pragma omp distribute simd reduction(- : x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(& : x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(| : x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(^ : x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(&& : x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(|| : x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(max : x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd reduction(min : x)
  for (i = 0; i < 16; ++i)
    ;
  struct X {
    int x;
  };
  struct X X;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected variable name}}
#pragma omp distribute simd reduction(+ : X.x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected variable name}}
#pragma omp distribute simd reduction(+ : x + x)
  for (i = 0; i < 16; ++i)
    ;
}

void test_loop_messages(void) {
  float a[100], b[100], c[100];
#pragma omp target
#pragma omp teams
// expected-error@+2 {{variable must be of integer or pointer type}}
#pragma omp distribute simd
  for (float fi = 0; fi < 10.0; fi++) {
    c[(int)fi] = a[(int)fi] + b[(int)fi];
  }
#pragma omp target
#pragma omp teams
// expected-error@+2 {{variable must be of integer or pointer type}}
#pragma omp distribute simd
  for (double fi = 0; fi < 10.0; fi++) {
    c[(int)fi] = a[(int)fi] + b[(int)fi];
  }
}

void linear_modifiers(int argc) {
  int k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd linear(k)
  for (k = 0; k < argc; ++k) ++k;
#pragma omp target
#pragma omp teams
  // omp60-error@+1 {{old syntax 'linear-modifier(list)' on 'linear' clause was deprecated, use new syntax 'linear(list: [linear-modifier,] step(step-size))'}}
#pragma omp distribute simd linear(val(k))
  for (k = 0; k < argc; ++k) ++k;
#pragma omp target
#pragma omp teams
  // omp60-error@+2 {{old syntax 'linear-modifier(list)' on 'linear' clause was deprecated, use new syntax 'linear(list: [linear-modifier,] step(step-size))'}}
  // expected-error@+1 {{expected 'val' modifier}}
#pragma omp distribute simd linear(uval(k))
  for (k = 0; k < argc; ++k) ++k;
#pragma omp target
#pragma omp teams
// omp60-error@+2 {{old syntax 'linear-modifier(list)' on 'linear' clause was deprecated, use new syntax 'linear(list: [linear-modifier,] step(step-size))'}}
// expected-error@+1 {{expected 'val' modifier}}
#pragma omp distribute simd linear(ref(k))
  for (k = 0; k < argc; ++k) ++k;
#pragma omp target
#pragma omp teams
// omp60-error@+2 {{old syntax 'linear-modifier(list)' on 'linear' clause was deprecated, use new syntax 'linear(list: [linear-modifier,] step(step-size))'}}
// expected-error@+1 {{expected 'val' modifier}}
#pragma omp distribute simd linear(foo(k))
  for (k = 0; k < argc; ++k) ++k;
}

void test_nontemporal(void) {
  int i;
// omp45-error@+4 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// expected-error@+3 {{expected expression}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd nontemporal(
  for (i = 0; i < 16; ++i)
    ;
// omp45-error@+4 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// expected-error@+3 2 {{expected expression}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd nontemporal(,
  for (i = 0; i < 16; ++i)
    ;
// omp45-error@+2 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// expected-error@+1 2 {{expected expression}}
#pragma omp distribute simd nontemporal(, )
  for (i = 0; i < 16; ++i)
    ;
// omp45-error@+2 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd nontemporal()
  for (i = 0; i < 16; ++i)
    ;
// omp45-error@+2 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd nontemporal(int)
  for (i = 0; i < 16; ++i)
    ;
// omp45-error@+2 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// omp50-and-later-var-error@+1 {{expected variable name}}
#pragma omp distribute simd nontemporal(0)
  for (i = 0; i < 16; ++i)
    ;
// omp45-error@+2 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// expected-error@+1 {{use of undeclared identifier 'x'}}
#pragma omp distribute simd nontemporal(x)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+3 {{use of undeclared identifier 'x'}}
// omp45-error@+2 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// expected-error@+1 {{use of undeclared identifier 'y'}}
#pragma omp distribute simd nontemporal(x, y)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+4 {{use of undeclared identifier 'x'}}
// expected-error@+3 {{use of undeclared identifier 'y'}}
// omp45-error@+2 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// expected-error@+1 {{use of undeclared identifier 'z'}}
#pragma omp distribute simd nontemporal(x, y, z)
  for (i = 0; i < 16; ++i)
    ;

  int x, y;
// omp45-error@+4 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// expected-error@+3 {{expected ',' or ')' in 'nontemporal' clause}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
#pragma omp distribute simd nontemporal(x :)
  for (i = 0; i < 16; ++i)
    ;
// omp45-error@+4 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// expected-error@+3 {{expected ')'}}
// expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected ',' or ')' in 'nontemporal' clause}}
#pragma omp distribute simd nontemporal(x :, )
  for (i = 0; i < 16; ++i)
    ;

// omp50-temporal-note@+3 {{defined as nontemporal}}
// omp-50-and-later-temporal-error@+2 {{a variable cannot appear in more than one nontemporal clause}}
// omp45-error@+1 2 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd nontemporal(x) nontemporal(x)
  for (i = 0; i < 16; ++i)
    ;

// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd private(x) nontemporal(x)
  for (i = 0; i < 16; ++i)
    ;

// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd nontemporal(x) private(x)
  for (i = 0; i < 16; ++i)
    ;

// omp45-error@+3 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
// expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected ',' or ')' in 'nontemporal' clause}} expected-error@+1 {{expected ')'}}
#pragma omp distribute simd nontemporal(x, y : 0)
  for (i = 0; i < 16; ++i)
    ;

// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd nontemporal(x) lastprivate(x)
  for (i = 0; i < 16; ++i)
    ;

// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd lastprivate(x) nontemporal(x)
  for (i = 0; i < 16; ++i)
    ;
// omp45-error@+2 {{unexpected OpenMP clause 'order' in directive '#pragma omp distribute simd'}}
// expected-error@+1 {{expected '(' after 'order'}}
#pragma omp distribute simd order
  for (int i = 0; i < 10; ++i)
    ;
// omp-50-and-later-error@+4 {{expected 'concurrent' in OpenMP clause 'order'}}
// expected-note@+3 {{to match this '('}}
// expected-error@+2 {{expected ')'}}
// omp45-error@+1 {{unexpected OpenMP clause 'order' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd order(
  for (int i = 0; i < 10; ++i)
    ;
// omp-50-and-later-error@+4 {{expected 'concurrent' in OpenMP clause 'order'}}
// expected-note@+3 {{to match this '('}}
// expected-error@+2 {{expected ')'}} 
// omp45-error@+1 {{unexpected OpenMP clause 'order' in directive '#pragma omp distribute simd'}} 
#pragma omp distribute simd order(none
  for (int i = 0; i < 10; ++i)
    ;
// expected-note@+3 {{to match this '('}}
// expected-error@+2 {{expected ')'}}
// omp45-error@+1 {{unexpected OpenMP clause 'order' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd order(concurrent
  for (int i = 0; i < 10; ++i)
    ;
// omp45-error@+1 {{unexpected OpenMP clause 'order' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd order(concurrent)
  for (int i = 0; i < 10; ++i)
    ;
  // omp-50-and-later-error@+2 {{expected 'concurrent' in OpenMP clause 'order'}}
  // omp45-error@+1 {{unexpected OpenMP clause 'order' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd order(unconstrained:)
  for (int i = 0; i < 10; ++i)
    ;
  // omp50-error@+4 {{expected 'concurrent' in OpenMP clause 'order'}}
  // expected-note@+3 {{to match this '('}}
  // expected-error@+2 {{expected ')'}}
  // omp45-error@+1 {{unexpected OpenMP clause 'order' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd order(reproducible:concurrent
  for (int i = 0; i < 10; ++i)
    ;
  // omp50-error@+2 {{expected 'concurrent' in OpenMP clause 'order'}}
  // omp45-error@+1 {{unexpected OpenMP clause 'order' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd order(reproducible:concurrent)
  for (int i = 0; i < 10; ++i)
    ;
  // omp50-error@+2 {{expected 'concurrent' in OpenMP clause 'order'}}
  // omp45-error@+1 {{unexpected OpenMP clause 'order' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd order(unconstrained:concurrent)
  for (int i = 0; i < 10; ++i)
    ;
  // omp-clause-error@+2 {{directive '#pragma omp distribute simd' cannot contain more than one 'order' clause}}
  // omp45-error@+1 2 {{unexpected OpenMP clause 'order' in directive '#pragma omp distribute simd'}}
#pragma omp distribute simd order(concurrent) order(concurrent)
  for (int i = 0; i < 10; ++i)
    ;
}
