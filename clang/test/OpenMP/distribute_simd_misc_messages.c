// RUN: %clang_cc1 -fsyntax-only -fopenmp -verify %s

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp distribute simd'}}
#pragma omp distribute simd

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp distribute simd'}}
#pragma omp distribute simd foo

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp distribute simd'}}
#pragma omp distribute simd safelen(4)

void test_no_clause() {
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

void test_branch_protected_scope() {
  int i = 0;
L1:
  ++i;

  int x[24];

#pragma omp target
#pragma omp teams
#pragma omp distribute simd
  for (i = 0; i < 16; ++i) {
    if (i == 5)
      goto L1; // expected-error {{use of undeclared label 'L1'}}
    else if (i == 6)
      return; // expected-error {{cannot return from OpenMP region}}
    else if (i == 7)
      goto L2;
    else if (i == 8) {
    L2:
      x[i]++;
    }
  }

  if (x[0] == 0)
    goto L2; // expected-error {{use of undeclared label 'L2'}}
  else if (x[1] == 1)
    goto L1;
}

void test_invalid_clause() {
  int i;
#pragma omp target
#pragma omp teams
// expected-warning@+1 {{extra tokens at the end of '#pragma omp distribute simd' are ignored}}
#pragma omp distribute simd foo bar
  for (i = 0; i < 16; ++i)
    ;
}

void test_non_identifiers() {
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

extern int foo();
void test_safelen() {
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
// expected-error@+1 {{expression is not an integer constant expression}}
#pragma omp distribute simd safelen(2.5)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expression is not an integer constant expression}}
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

void test_simdlen() {
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
// expected-error@+1 {{expression is not an integer constant expression}}
#pragma omp distribute simd simdlen(2.5)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expression is not an integer constant expression}}
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

void test_safelen_simdlen() {
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

void test_collapse() {
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
    ; // expected-error {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4,
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4, )
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
#pragma omp target
#pragma omp teams
// xxpected-error@+1 {{expected expression}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4)
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4 4)
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4, , 4)
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
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
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp distribute simd collapse(4, 8)
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp distribute simd', but found only 1}}
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expression is not an integer constant expression}}
#pragma omp distribute simd collapse(2.5)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expression is not an integer constant expression}}
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
// expected-note@+3 {{defined as reduction}}
#pragma omp target
#pragma omp teams
#pragma omp distribute simd collapse(2) reduction(+ : i)
  for (i = 0; i < 16; ++i)
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

void test_linear() {
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

  int x, y;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd linear(x :)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd linear(x :, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd linear(x : 1)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd linear(x : 2 * 2)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd linear(x : 1, y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd linear(x : 1, y, z : 1)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-note@+2 {{defined as linear}}
// expected-error@+1 {{linear variable cannot be linear}}
#pragma omp distribute simd linear(x) linear(x)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-note@+2 {{defined as private}}
// expected-error@+1 {{private variable cannot be linear}}
#pragma omp distribute simd private(x) linear(x)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-note@+2 {{defined as linear}}
// expected-error@+1 {{linear variable cannot be private}}
#pragma omp distribute simd linear(x) private(x)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-warning@+1 {{zero linear step (x and other variables in clause should probably be const)}}
#pragma omp distribute simd linear(x, y : 0)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-note@+2 {{defined as linear}}
// expected-error@+1 {{linear variable cannot be lastprivate}}
#pragma omp distribute simd linear(x) lastprivate(x)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-note@+2 {{defined as lastprivate}}
// expected-error@+1 {{lastprivate variable cannot be linear}}
#pragma omp distribute simd lastprivate(x) linear(x)
  for (i = 0; i < 16; ++i)
    ;
}

void test_aligned() {
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

void test_private() {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected expression}}
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp distribute simd private(
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
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

void test_firstprivate() {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd firstprivate(
  for (i = 0; i < 16; ++i)
    ;
}

void test_lastprivate() {
  int i;
#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp distribute simd lastprivate(
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
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

void test_reduction() {
  int i, x, y;
#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected ')'}} expected-note@+3 {{to match this '('}}
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
// expected-error@+3 {{expected ')'}} expected-note@+3 {{to match this '('}}
// expected-error@+2 {{expected identifier}}
// expected-warning@+1 {{missing ':' after reduction identifier - ignoring}}
#pragma omp distribute simd reduction(,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected ')'}} expected-note@+3 {{to match this '('}}
// expected-error@+2 {{expected expression}}
// expected-warning@+1 {{missing ':' after reduction identifier - ignoring}}
#pragma omp distribute simd reduction(+
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
#pragma omp teams
// expected-error@+3 {{expected ')'}} expected-note@+3 {{to match this '('}}
//
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

void test_loop_messages() {
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
  int f;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd linear(f)
  for (int k = 0; k < argc; ++k) ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd linear(val(f))
  for (int k = 0; k < argc; ++k) ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd linear(uval(f)) // expected-error {{expected 'val' modifier}}
  for (int k = 0; k < argc; ++k) ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd linear(ref(f)) // expected-error {{expected 'val' modifier}}
  for (int k = 0; k < argc; ++k) ++k;
#pragma omp target
#pragma omp teams
#pragma omp distribute simd linear(foo(f)) // expected-error {{expected 'val' modifier}}
  for (int k = 0; k < argc; ++k) ++k;
}

