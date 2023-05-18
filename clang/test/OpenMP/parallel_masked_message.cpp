// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp parallel masked
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

#pragma omp parallel masked // expected-error {{unexpected OpenMP directive '#pragma omp parallel masked'}}

int foo() { 
  return 0;
}

int a;
struct S;
S& bar();
int main(int argc, char **argv) {
  #pragma omp parallel masked nowait // expected-error {{unexpected OpenMP clause 'nowait' in directive '#pragma omp parallel masked'}}
  #pragma omp parallel masked unknown // expected-warning {{extra tokens at the end of '#pragma omp parallel masked' are ignored}}
  foo();
  {
    #pragma omp masked
  } // expected-error {{expected statement}}
  {
    #pragma omp parallel masked
  } // expected-error {{expected statement}}

  S &s = bar();
  int tid = 0;
  #pragma omp parallel masked filter(tid)
  (void)&s;
  #pragma omp parallel masked { // expected-warning {{extra tokens at the end of '#pragma omp parallel masked' are ignored}}
  foo();
  #pragma omp parallel masked ( // expected-warning {{extra tokens at the end of '#pragma omp parallel masked' are ignored}}
  foo();
  #pragma omp parallel masked [ // expected-warning {{extra tokens at the end of '#pragma omp parallel masked' are ignored}}
  foo();
  #pragma omp parallel masked ] // expected-warning {{extra tokens at the end of '#pragma omp parallel masked' are ignored}}
  foo();
  #pragma omp parallel masked ) // expected-warning {{extra tokens at the end of '#pragma omp parallel masked' are ignored}}
  foo();
  #pragma omp parallel masked } // expected-warning {{extra tokens at the end of '#pragma omp parallel masked' are ignored}}
  foo();
  #pragma omp parallel masked
  // expected-warning@+1 {{extra tokens at the end of '#pragma omp parallel masked' are ignored}}
  #pragma omp parallel masked unknown()
  foo();
  L1:
    foo();
  #pragma omp parallel masked
  ;
  #pragma omp parallel masked
  {

  for (int i = 0; i < 10; ++i) {
    switch(argc) {
     case (0):
      #pragma omp parallel masked
      {
        foo();
        break; // expected-error {{'break' statement not in loop or switch statement}}
        continue; // expected-error {{'continue' statement not in loop statement}}
      }
      default:
       break;
    }
  }
    goto L1; // expected-error {{use of undeclared label 'L1'}}
    argc++;
  }
#pragma omp parallel masked default(none) // expected-note 2 {{explicit data sharing attribute requested here}}
  {
    ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
    ++a;    // expected-error {{variable 'a' must have explicitly specified data sharing attributes}}
  }

  goto L2; // expected-error {{use of undeclared label 'L2'}}
  #pragma omp parallel masked
  L2:
  foo();
  #pragma omp parallel masked
  {
    return 1; // expected-error {{cannot return from OpenMP region}}
  }
  return 0;
}
