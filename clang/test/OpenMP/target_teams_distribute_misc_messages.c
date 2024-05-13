// RUN: %clang_cc1 -fsyntax-only -fopenmp -verify %s -Wuninitialized

// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -verify %s -Wuninitialized

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp target teams distribute'}}
#pragma omp target teams distribute

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp target teams distribute'}}
#pragma omp target teams distribute foo

void test_no_clause(void) {
  int i;
#pragma omp target teams distribute
  for (i = 0; i < 16; ++i)
    ;

// expected-error@+2 {{statement after '#pragma omp target teams distribute' must be a for loop}}
#pragma omp target teams distribute
  ++i;
}

void test_branch_protected_scope(void) {
  int i = 0;
L1:
  ++i;

  int x[24];

#pragma omp target teams distribute
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

void test_invalid_clause(void) {
  int i;
// expected-warning@+1 {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
#pragma omp target teams distribute foo bar
  for (i = 0; i < 16; ++i)
    ;
}

void test_non_identifiers(void) {
  int i, x;

// expected-warning@+1 {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
#pragma omp target teams distribute;
  for (i = 0; i < 16; ++i)
    ;

// expected-warning@+1 {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
#pragma omp target teams distribute private(x);
  for (i = 0; i < 16; ++i)
    ;

// expected-warning@+1 {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
#pragma omp target teams distribute, private(x);
  for (i = 0; i < 16; ++i)
    ;
}

extern int foo(void);

void test_collapse(void) {
  int i;
// expected-error@+1 {{expected '('}}
#pragma omp target teams distribute collapse
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute collapse(
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected expression}}
#pragma omp target teams distribute collapse()
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute collapse(,
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected expression}}  expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute collapse(, )
  for (i = 0; i < 16; ++i)
    ;
// expected-warning@+2 {{extra tokens at the end of '#pragma omp target teams distribute' are ignored}}
// expected-error@+1 {{expected '('}}
#pragma omp target teams distribute collapse 4)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp target teams distribute collapse(4
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp target teams distribute', but found only 1}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp target teams distribute collapse(4,
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp target teams distribute', but found only 1}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp target teams distribute collapse(4, )
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp target teams distribute', but found only 1}}
// expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp target teams distribute collapse(4)
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp target teams distribute', but found only 1}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp target teams distribute collapse(4 4)
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp target teams distribute', but found only 1}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp target teams distribute collapse(4, , 4)
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp target teams distribute', but found only 1}}
#pragma omp target teams distribute collapse(4)
  for (int i1 = 0; i1 < 16; ++i1)
    for (int i2 = 0; i2 < 16; ++i2)
      for (int i3 = 0; i3 < 16; ++i3)
        for (int i4 = 0; i4 < 16; ++i4)
          foo();
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}} expected-note@+1 {{as specified in 'collapse' clause}}
#pragma omp target teams distribute collapse(4, 8)
  for (i = 0; i < 16; ++i)
    ; // expected-error {{expected 4 for loops after '#pragma omp target teams distribute', but found only 1}}
// expected-error@+1 {{integer constant expression}}
#pragma omp target teams distribute collapse(2.5)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{integer constant expression}}
#pragma omp target teams distribute collapse(foo())
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{argument to 'collapse' clause must be a strictly positive integer value}}
#pragma omp target teams distribute collapse(-5)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{argument to 'collapse' clause must be a strictly positive integer value}}
#pragma omp target teams distribute collapse(0)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{argument to 'collapse' clause must be a strictly positive integer value}}
#pragma omp target teams distribute collapse(5 - 5)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+3 {{loop iteration variable in the associated loop of 'omp target teams distribute' directive may not be firstprivate, predetermined as private}}
// expected-note@+1 {{defined as firstprivate}}
#pragma omp target teams distribute collapse(2) firstprivate(i)
  for (i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
#pragma omp parallel for reduction(+ : i, j)
      for (int k = 0; k < 16; ++k)
        i += j;
}

void test_private(void) {
  int i;
// expected-error@+2 {{expected expression}}
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp target teams distribute private(
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp target teams distribute private(,
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 2 {{expected expression}}
#pragma omp target teams distribute private(, )
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected expression}}
#pragma omp target teams distribute private()
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected expression}}
#pragma omp target teams distribute private(int)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected variable name}}
#pragma omp target teams distribute private(0)
  for (i = 0; i < 16; ++i)
    ;

  int x, y, z;
#pragma omp target teams distribute private(x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target teams distribute private(x, y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target teams distribute private(x, y, z)
  for (i = 0; i < 16; ++i) {
    x = y * i + z;
  }
}

void test_lastprivate(void) {
  int i;
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp target teams distribute lastprivate(
  for (i = 0; i < 16; ++i)
    ;

// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp target teams distribute lastprivate(,
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 2 {{expected expression}}
#pragma omp target teams distribute lastprivate(, )
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected expression}}
#pragma omp target teams distribute lastprivate()
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected expression}}
#pragma omp target teams distribute lastprivate(int)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected variable name}}
#pragma omp target teams distribute lastprivate(0)
  for (i = 0; i < 16; ++i)
    ;

  int x, y, z;
#pragma omp target teams distribute lastprivate(x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target teams distribute lastprivate(x, y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target teams distribute lastprivate(x, y, z)
  for (i = 0; i < 16; ++i)
    ;
}

void test_firstprivate(void) {
  int i;
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp target teams distribute firstprivate(
  for (i = 0; i < 16; ++i)
    ;

// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp target teams distribute firstprivate(,
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 2 {{expected expression}}
#pragma omp target teams distribute firstprivate(, )
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected expression}}
#pragma omp target teams distribute firstprivate()
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected expression}}
#pragma omp target teams distribute firstprivate(int)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 {{expected variable name}}
#pragma omp target teams distribute firstprivate(0)
  for (i = 0; i < 16; ++i)
    ;

  int x, y, z;
// expected-error@+1 {{lastprivate variable cannot be firstprivate}} expected-note@+1 {{defined as lastprivate}}
#pragma omp target teams distribute lastprivate(x) firstprivate(x)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 2 {{lastprivate variable cannot be firstprivate}} expected-note@+1 2 {{defined as lastprivate}}
#pragma omp target teams distribute lastprivate(x, y) firstprivate(x, y)
  for (i = 0; i < 16; ++i)
    ;
// expected-error@+1 3 {{lastprivate variable cannot be firstprivate}} expected-note@+1 3 {{defined as lastprivate}}
#pragma omp target teams distribute lastprivate(x, y, z) firstprivate(x, y, z)
  for (i = 0; i < 16; ++i)
    ;
}

void test_loop_messages(void) {
  float a[100], b[100], c[100];
// expected-error@+2 {{variable must be of integer or pointer type}}
#pragma omp target teams distribute
  for (float fi = 0; fi < 10.0; fi++) {
    c[(int)fi] = a[(int)fi] + b[(int)fi];
  }
// expected-error@+2 {{variable must be of integer or pointer type}}
#pragma omp target teams distribute
  for (double fi = 0; fi < 10.0; fi++) {
    c[(int)fi] = a[(int)fi] + b[(int)fi];
  }
}

