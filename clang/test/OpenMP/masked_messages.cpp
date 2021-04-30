// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp masked
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

void yyy(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp masked filter(1)
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

int foo();

int main() {
  #pragma omp masked
  ;
  #pragma omp masked filter(1) filter(2) // expected-error {{directive '#pragma omp masked' cannot contain more than one 'filter' clause}}
  ;
  int x,y,z;
  #pragma omp masked filter(x) filter(y) filter(z) // expected-error 2 {{directive '#pragma omp masked' cannot contain more than one 'filter' clause}}
  ;
  #pragma omp masked nowait // expected-error {{unexpected OpenMP clause 'nowait' in directive '#pragma omp masked'}}
  #pragma omp masked unknown // expected-warning {{extra tokens at the end of '#pragma omp masked' are ignored}}
  foo();
  {
    #pragma omp masked
  } // expected-error {{expected statement}}
  {
    #pragma omp masked filter(2)
  } // expected-error {{expected statement}}
  #pragma omp for
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp masked filter(1) // expected-error {{region cannot be closely nested inside 'for' region}}
    foo();
  }
  #pragma omp sections
  {
    foo();
    #pragma omp masked // expected-error {{region cannot be closely nested inside 'sections' region}}
    foo();
  }
  #pragma omp single
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp masked allocate(i) // expected-error {{region cannot be closely nested inside 'single' region}} expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp masked'}}
    foo();
  }
  #pragma omp masked
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp masked
    foo();
  }
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i)
  #pragma omp masked // expected-error {{region cannot be closely nested inside 'for' region}}
  {
    foo();
  }

  return 0;
}

int foo() {
  L1: // expected-note {{jump exits scope of OpenMP structured block}}
    foo();
  #pragma omp masked filter(0)
  {
    foo();
    goto L1; // expected-error {{cannot jump from this goto statement to its label}}
  }
  goto L2; // expected-error {{cannot jump from this goto statement to its label}}
  #pragma omp masked filter(-2)
  { // expected-note {{jump bypasses OpenMP structured block}}
    L2:
    foo();
  }

  return 0;
}
