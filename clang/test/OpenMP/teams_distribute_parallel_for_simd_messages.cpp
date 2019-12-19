// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -verify=expected,omp45 -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -verify=expected,omp50 -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -verify=expected,omp45 -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -verify=expected,omp50 -std=c++11 %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i = 0; i < 10; ++i)
    argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

void foo() {
}

static int pvt;
#pragma omp threadprivate(pvt)

#pragma omp teams distribute parallel for simd // expected-error {{unexpected OpenMP directive '#pragma omp teams distribute parallel for simd'}}

int main(int argc, char **argv) {
  #pragma omp target
  #pragma omp teams distribute parallel for simd
  f; // expected-error {{use of undeclared identifier 'f'}}
#pragma omp target
#pragma omp teams distribute parallel for simd { // expected-warning {{extra tokens at the end of '#pragma omp teams distribute parallel for simd' are ignored}}
  for (int i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams distribute parallel for simd ( // expected-warning {{extra tokens at the end of '#pragma omp teams distribute parallel for simd' are ignored}}
  for (int i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams distribute parallel for simd[ // expected-warning {{extra tokens at the end of '#pragma omp teams distribute parallel for simd' are ignored}}
  for (int i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams distribute parallel for simd] // expected-warning {{extra tokens at the end of '#pragma omp teams distribute parallel for simd' are ignored}}
  for (int i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams distribute parallel for simd) // expected-warning {{extra tokens at the end of '#pragma omp teams distribute parallel for simd' are ignored}}
  for (int i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams distribute parallel for simd } // expected-warning {{extra tokens at the end of '#pragma omp teams distribute parallel for simd' are ignored}}
  for (int i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i = 0; i < argc; ++i)
    foo();
// expected-warning@+2 {{extra tokens at the end of '#pragma omp teams distribute parallel for simd' are ignored}}
#pragma omp target
#pragma omp teams distribute parallel for simd unknown()
  for (int i = 0; i < argc; ++i)
    foo();
L1:
  for (int i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i = 0; i < argc; ++i)
    foo();
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i = 0; i < argc; ++i) {
    goto L1; // expected-error {{use of undeclared label 'L1'}}
    argc++;
  }

  for (int i = 0; i < 10; ++i) {
    switch (argc) {
    case (0):
#pragma omp target
#pragma omp teams distribute parallel for simd
      for (int i = 0; i < argc; ++i) {
        foo();
        break; // expected-error {{'break' statement cannot be used in OpenMP for loop}}
        continue;
      }
    default:
      break;
    }
  }
#pragma omp target
#pragma omp teams distribute parallel for simd default(none) // expected-note {{explicit data sharing attribute requested here}}
  for (int i = 0; i < 10; ++i)
    ++argc; // expected-error {{ariable 'argc' must have explicitly specified data sharing attributes}}

  goto L2; // expected-error {{use of undeclared label 'L2'}}
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i = 0; i < argc; ++i)
  L2:
  foo();
#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i = 0; i < argc; ++i) {
    return 1; // expected-error {{cannot return from OpenMP region}}
  }

  [[]] // expected-error {{an attribute list cannot appear here}}
#pragma omp target
#pragma omp teams distribute parallel for simd
      for (int n = 0; n < 100; ++n) {
  }

#pragma omp target
#pragma omp teams distribute parallel for simd copyin(pvt) // expected-error {{unexpected OpenMP clause 'copyin' in directive '#pragma omp teams distribute parallel for simd'}}
  for (int n = 0; n < 100; ++n) {}

  return 0;
}

void test_ordered() {
#pragma omp target
#pragma omp teams distribute parallel for simd ordered // expected-error {{unexpected OpenMP clause 'ordered' in directive '#pragma omp teams distribute parallel for simd'}}
  for (int i = 0; i < 16; ++i)
    ;
}

void test_nontemporal() {
  int i;
#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} expected-error@+1 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp teams distribute parallel for simd nontemporal(
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} expected-error@+1 2 {{expected expression}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp teams distribute parallel for simd nontemporal(,
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} expected-error@+1 2 {{expected expression}}
#pragma omp teams distribute parallel for simd nontemporal(, )
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} expected-error@+1 {{expected expression}}
#pragma omp teams distribute parallel for simd nontemporal()
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} expected-error@+1 {{expected '(' for function-style cast or type construction}}
#pragma omp teams distribute parallel for simd nontemporal(int)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} omp50-error@+1 {{expected variable name}}
#pragma omp teams distribute parallel for simd nontemporal(0)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} expected-error@+1 {{use of undeclared identifier 'x'}}
#pragma omp teams distribute parallel for simd nontemporal(x)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
// expected-error@+2 {{use of undeclared identifier 'x'}}
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} expected-error@+1 {{use of undeclared identifier 'y'}}
#pragma omp teams distribute parallel for simd nontemporal(x, y)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
// expected-error@+3 {{use of undeclared identifier 'x'}}
// expected-error@+2 {{use of undeclared identifier 'y'}}
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} expected-error@+1 {{use of undeclared identifier 'z'}}
#pragma omp teams distribute parallel for simd nontemporal(x, y, z)
  for (i = 0; i < 16; ++i)
    ;

  int x, y;
#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} expected-error@+1 {{expected ',' or ')' in 'nontemporal' clause}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
#pragma omp teams distribute parallel for simd nontemporal(x :)
  for (i = 0; i < 16; ++i)
    ;
#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}} expected-error@+1 {{expected ',' or ')' in 'nontemporal' clause}}
#pragma omp teams distribute parallel for simd nontemporal(x :, )
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
// omp50-note@+2 {{defined as nontemporal}}
// omp45-error@+1 2 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} omp50-error@+1 {{a variable cannot appear in more than one nontemporal clause}}
#pragma omp teams distribute parallel for simd nontemporal(x) nontemporal(x)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}}
#pragma omp teams distribute parallel for simd private(x) nontemporal(x)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}}
#pragma omp teams distribute parallel for simd nontemporal(x) private(x)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}} expected-note@+1 {{to match this '('}} expected-error@+1 {{expected ',' or ')' in 'nontemporal' clause}} expected-error@+1 {{expected ')'}}
#pragma omp teams distribute parallel for simd nontemporal(x, y : 0)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}}
#pragma omp teams distribute parallel for simd nontemporal(x) lastprivate(x)
  for (i = 0; i < 16; ++i)
    ;

#pragma omp target
// omp45-error@+1 {{unexpected OpenMP clause 'nontemporal' in directive '#pragma omp teams distribute parallel for simd'}}
#pragma omp teams distribute parallel for simd lastprivate(x) nontemporal(x)
  for (i = 0; i < 16; ++i)
    ;
}

