// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++11 -o - %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++11 -o - %s

void foo() {
}

#pragma omp parallel sections // expected-error {{unexpected OpenMP directive '#pragma omp parallel sections'}}

int main(int argc, char **argv) {
#pragma omp parallel sections {// expected-warning {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
  {
    foo();
  }
#pragma omp parallel sections( // expected-warning {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
  {
    foo();
  }
#pragma omp parallel sections[ // expected-warning {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
  {
    foo();
  }
#pragma omp parallel sections] // expected-warning {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
  {
    foo();
  }
#pragma omp parallel sections) // expected-warning {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
  {
    foo();
  }
#pragma omp parallel sections } // expected-warning {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
  {
    foo();
  }
// expected-warning@+1 {{extra tokens at the end of '#pragma omp parallel sections' are ignored}}
#pragma omp parallel sections unknown()
  {
    foo();
#pragma omp section
  L1:
    foo();
  }
#pragma omp parallel sections
  {
    ;
  }
#pragma omp parallel sections
  {
    goto L1; // expected-error {{use of undeclared label 'L1'}}
  }

  for (int i = 0; i < 10; ++i) {
    switch (argc) {
    case (0):
#pragma omp parallel sections
    {
      foo();
      break;    // expected-error {{'break' statement not in loop or switch statement}}
      continue; // expected-error {{'continue' statement not in loop statement}}
    }
    default:
      break;
    }
  }
#pragma omp parallel sections default(none) // expected-note {{explicit data sharing attribute requested here}}
  {
    ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
  }

  goto L2; // expected-error {{use of undeclared label 'L2'}}
#pragma omp parallel sections
  {
  L2:
    foo();
  }
#pragma omp parallel sections
  {
    return 1; // expected-error {{cannot return from OpenMP region}}
  }

  [[]] // expected-error {{an attribute list cannot appear here}}
#pragma omp parallel sections
  {
  }

  return 0;
}
