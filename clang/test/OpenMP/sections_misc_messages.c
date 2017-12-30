// RUN: %clang_cc1 -fsyntax-only -fopenmp -verify %s

// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -verify %s

void foo();

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp sections'}}
#pragma omp sections

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp sections'}}
#pragma omp sections foo

void test_no_clause() {
  int i;
#pragma omp sections
  {
    foo();
  }

// expected-error@+2 {{the statement for '#pragma omp sections' must be a compound statement}}
#pragma omp sections
  ++i;

#pragma omp sections
  {
    foo();
    foo(); // expected-error {{statement in 'omp sections' directive must be enclosed into a section region}}
  }
}

void test_branch_protected_scope() {
  int i = 0;
L1:
  ++i;

  int x[24];

#pragma omp parallel
#pragma omp sections
  {
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
#pragma omp section
    if (i == 5)
      goto L1; // expected-error {{use of undeclared label 'L1'}}
    else if (i == 6)
      return; // expected-error {{cannot return from OpenMP region}}
    else if (i == 7)
      goto L3;
    else if (i == 8) {
    L3:
      x[i]++;
    }
  }

  if (x[0] == 0)
    goto L2; // expected-error {{use of undeclared label 'L2'}}
  else if (x[1] == 1)
    goto L1;
  goto L3; // expected-error {{use of undeclared label 'L3'}}
}

void test_invalid_clause() {
  int i;
#pragma omp parallel
// expected-warning@+1 {{extra tokens at the end of '#pragma omp sections' are ignored}}
#pragma omp sections foo bar
  {
    foo();
// expected-error@+1 {{unexpected OpenMP clause 'nowait' in directive '#pragma omp section'}}
#pragma omp section nowait
    ;
  }
}

void test_non_identifiers() {
  int i, x;

#pragma omp parallel
// expected-warning@+1 {{extra tokens at the end of '#pragma omp sections' are ignored}}
#pragma omp sections;
  {
    foo();
  }
#pragma omp parallel
// expected-error@+2 {{unexpected OpenMP clause 'linear' in directive '#pragma omp sections'}}
// expected-warning@+1 {{extra tokens at the end of '#pragma omp sections' are ignored}}
#pragma omp sections linear(x);
  {
    foo();
  }

#pragma omp parallel
// expected-warning@+1 {{extra tokens at the end of '#pragma omp sections' are ignored}}
#pragma omp sections private(x);
  {
    foo();
  }

#pragma omp parallel
// expected-warning@+1 {{extra tokens at the end of '#pragma omp sections' are ignored}}
#pragma omp sections, private(x);
  {
    foo();
  }
}

void test_private() {
  int i;
#pragma omp parallel
// expected-error@+2 {{expected expression}}
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp sections private(
  {
    foo();
  }
#pragma omp parallel
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp sections private(,
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 2 {{expected expression}}
#pragma omp sections private(, )
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 {{expected expression}}
#pragma omp sections private()
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 {{expected expression}}
#pragma omp sections private(int)
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 {{expected variable name}}
#pragma omp sections private(0)
  {
    foo();
  }

  int x, y, z;
#pragma omp parallel
#pragma omp sections private(x)
  {
    foo();
  }
#pragma omp parallel
#pragma omp sections private(x, y)
  {
    foo();
  }
#pragma omp parallel
#pragma omp sections private(x, y, z)
  {
    foo();
  }
}

void test_lastprivate() {
  int i;
#pragma omp parallel
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp sections lastprivate(
  {
    foo();
  }

#pragma omp parallel
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp sections lastprivate(,
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 2 {{expected expression}}
#pragma omp sections lastprivate(, )
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 {{expected expression}}
#pragma omp sections lastprivate()
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 {{expected expression}}
#pragma omp sections lastprivate(int)
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 {{expected variable name}}
#pragma omp sections lastprivate(0)
  {
    foo();
  }

  int x, y, z;
#pragma omp parallel
#pragma omp sections lastprivate(x)
  {
    foo();
  }
#pragma omp parallel
#pragma omp sections lastprivate(x, y)
  {
    foo();
  }
#pragma omp parallel
#pragma omp sections lastprivate(x, y, z)
  {
    foo();
  }
}

void test_firstprivate() {
  int i;
#pragma omp parallel
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp sections firstprivate(
  {
    foo();
  }

#pragma omp parallel
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp sections firstprivate(,
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 2 {{expected expression}}
#pragma omp sections firstprivate(, )
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 {{expected expression}}
#pragma omp sections firstprivate()
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 {{expected expression}}
#pragma omp sections firstprivate(int)
  {
    foo();
  }
#pragma omp parallel
// expected-error@+1 {{expected variable name}}
#pragma omp sections firstprivate(0)
  {
    foo();
  }

  int x, y, z;
#pragma omp parallel
#pragma omp sections lastprivate(x) firstprivate(x)
  {
    foo();
  }
#pragma omp parallel
#pragma omp sections lastprivate(x, y) firstprivate(x, y)
  {
    foo();
  }
#pragma omp parallel
#pragma omp sections lastprivate(x, y, z) firstprivate(x, y, z)
  {
    foo();
  }
}

void test_nowait() {
#pragma omp parallel
#pragma omp sections nowait nowait // expected-error {{directive '#pragma omp sections' cannot contain more than one 'nowait' clause}}
  {
    ;
  }
}
