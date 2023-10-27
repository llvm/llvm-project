// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
  [[omp::directive(masked)]]
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

void yyy(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
  [[omp::directive(masked filter(1))]]
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

int foo();

int main() {
  [[omp::directive(masked)]]
  ;
  [[omp::directive(masked filter(1) filter(2))]] // expected-error {{directive '#pragma omp masked' cannot contain more than one 'filter' clause}}
  ;
  int x,y,z;
  [[omp::directive(masked filter(x) filter(y) filter(z))]] // expected-error 2 {{directive '#pragma omp masked' cannot contain more than one 'filter' clause}}
  ;
  [[omp::directive(masked nowait)]] // expected-error {{unexpected OpenMP clause 'nowait' in directive '#pragma omp masked'}}
  [[omp::directive(masked unknown)]] // expected-warning {{extra tokens at the end of '#pragma omp masked' are ignored}}
  foo();
  {
	[[omp::directive(masked)]]
  } // expected-error {{expected statement}}
  {
	[[omp::directive(masked filter(2))]]
  } // expected-error {{expected statement}}
  [[omp::directive(for)]]
  for (int i = 0; i < 10; ++i) {
    foo();
    [[omp::directive(masked filter(1))]] // expected-error {{region cannot be closely nested inside 'for' region}}
    foo();
  }
  [[omp::directive(sections)]]
  {
    foo();
    [[omp::directive(masked)]] // expected-error {{region cannot be closely nested inside 'sections' region}}
    foo();
  }
  [[omp::directive(single)]]
  for (int i = 0; i < 10; ++i) {
    foo();
    [[omp::directive(masked allocate(i))]] // expected-error {{region cannot be closely nested inside 'single' region}} expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp masked'}}
    foo();
  }
  [[omp::directive(masked)]]
  for (int i = 0; i < 10; ++i) {
    foo();
    [[omp::directive(masked)]]
    foo();
  }
  [[omp::directive(for ordered)]]
  for (int i = 0; i < 10; ++i)
  [[omp::directive(masked)]] // expected-error {{region cannot be closely nested inside 'for' region}}
  {
    foo();
  }

  return 0;
}

int foo() {
  L1: // expected-note {{jump exits scope of OpenMP structured block}}
    foo();
  [[omp::directive(masked filter(0))]]
  {
    foo();
    goto L1; // expected-error {{cannot jump from this goto statement to its label}}
  }
  goto L2; // expected-error {{cannot jump from this goto statement to its label}}
  [[omp::directive(masked filter(-2))]]
  { // expected-note {{jump bypasses OpenMP structured block}}
    L2:
    foo();
  }

  return 0;
}

