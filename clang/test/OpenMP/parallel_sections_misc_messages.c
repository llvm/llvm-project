// RUN: %clang_cc1 -fsyntax-only -fopenmp -verify %s -Wuninitialized

// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -verify %s -Wuninitialized

void foo();

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp parallel sections'}}
#pragma omp parallel sections

// expected-error@+1 {{unexpected OpenMP directive '#pragma omp parallel sections'}}
#pragma omp parallel sections foo

void test_no_clause() {
  int i;
#pragma omp parallel sections
  {
    foo();
  }

// expected-error@+2 {{the statement for '#pragma omp parallel sections' must be a compound statement}}
#pragma omp parallel sections
  ++i;

#pragma omp parallel sections
  {
    foo();
    foo(); // expected-error {{statement in 'omp parallel sections' directive must be enclosed into a section region}}
  }

}

void test_branch_protected_scope() {
  int i = 0;
L1:
  ++i;

  int x[24];

#pragma omp parallel sections
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
      goto L1;
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
// expected-warning@+1 {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
#pragma omp parallel sections foo bar
  {
    foo();
// expected-error@+1 {{unexpected OpenMP clause 'nowait' in directive '#pragma omp section'}}
#pragma omp section nowait
    ;
  }
}

void test_non_identifiers() {
  int i, x;

// expected-warning@+1 {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
#pragma omp parallel sections;
  {
    foo();
  }
// expected-error@+2 {{unexpected OpenMP clause 'linear' in directive '#pragma omp parallel sections'}}
// expected-warning@+1 {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
#pragma omp parallel sections linear(x);
  {
    foo();
  }

// expected-warning@+1 {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
#pragma omp parallel sections private(x);
  {
    foo();
  }

// expected-warning@+1 {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
#pragma omp parallel sections, private(x);
  {
    foo();
  }
}

void test_private() {
  int i;
// expected-error@+2 {{expected expression}}
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp parallel sections private(
  {
    foo();
  }
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp parallel sections private(,
  {
    foo();
  }
// expected-error@+1 2 {{expected expression}}
#pragma omp parallel sections private(, )
  {
    foo();
  }
// expected-error@+1 {{expected expression}}
#pragma omp parallel sections private()
  {
    foo();
  }
// expected-error@+1 {{expected expression}}
#pragma omp parallel sections private(int)
  {
    foo();
  }
// expected-error@+1 {{expected variable name}}
#pragma omp parallel sections private(0)
  {
    foo();
  }

  int x, y, z;
#pragma omp parallel sections private(x)
  {
    foo();
  }
#pragma omp parallel sections private(x, y)
  {
    foo();
  }
#pragma omp parallel sections private(x, y, z)
  {
    foo();
  }
}

void test_lastprivate() {
  int i;
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp parallel sections lastprivate(
  {
    foo();
  }

// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp parallel sections lastprivate(,
  {
    foo();
  }
// expected-error@+1 2 {{expected expression}}
#pragma omp parallel sections lastprivate(, )
  {
    foo();
  }
// expected-error@+1 {{expected expression}}
#pragma omp parallel sections lastprivate()
  {
    foo();
  }
// expected-error@+1 {{expected expression}}
#pragma omp parallel sections lastprivate(int)
  {
    foo();
  }
// expected-error@+1 {{expected variable name}}
#pragma omp parallel sections lastprivate(0)
  {
    foo();
  }

  int x, y, z;
#pragma omp parallel sections lastprivate(x)
  {
    foo();
  }
#pragma omp parallel sections lastprivate(x, y)
  {
    foo();
  }
#pragma omp parallel sections lastprivate(x, y, z)
  {
    foo();
  }
}

void test_firstprivate() {
  int i;
// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 {{expected expression}}
#pragma omp parallel sections firstprivate(
  {
    foo();
  }

// expected-error@+2 {{expected ')'}} expected-note@+2 {{to match this '('}}
// expected-error@+1 2 {{expected expression}}
#pragma omp parallel sections firstprivate(,
  {
    foo();
  }
// expected-error@+1 2 {{expected expression}}
#pragma omp parallel sections firstprivate(, )
  {
    foo();
  }
// expected-error@+1 {{expected expression}}
#pragma omp parallel sections firstprivate()
  {
    foo();
  }
// expected-error@+1 {{expected expression}}
#pragma omp parallel sections firstprivate(int)
  {
    foo();
  }
// expected-error@+1 {{expected variable name}}
#pragma omp parallel sections firstprivate(0)
  {
    foo();
  }

  int x, y, z;
#pragma omp parallel sections lastprivate(x) firstprivate(x)
  {
    foo();
  }
#pragma omp parallel sections lastprivate(x, y) firstprivate(x, y)
  {
    foo();
  }
#pragma omp parallel sections lastprivate(x, y, z) firstprivate(x, y, z)
  {
    foo();
  }
}

