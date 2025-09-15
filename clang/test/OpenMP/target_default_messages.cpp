
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -DOMP60 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=60 -DOMP60 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -DOMP52 %s -Wuninitialized

void foo();

namespace {
static int y = 0;
}
static int x = 0;

int main(int argc, char **argv) {
#ifdef OMP60
  #pragma omp target default // expected-error {{expected '(' after 'default'}}
  for (int i=0; i<200; i++) foo();
#pragma omp target  default( // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<200; i++) foo();
#pragma omp target default() // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
  for (int i=0; i<200; i++) foo();
  #pragma omp target default (none // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i=0; i<200; i++) foo();
#pragma omp target  default(x) // expected-error {{expected 'none', 'shared', 'private' or 'firstprivate' in OpenMP clause 'default'}}
  for (int i=0; i<200; i++) foo();
#endif 

#ifdef OMP52
#pragma omp target default(firstprivate) // expected-error {{unexpected OpenMP clause 'default' in directive '#pragma omp target'}}
  for (int i = 0; i < 200; i++) {
    ++x;
    ++y;
  }
#pragma omp target default(private) // expected-error {{unexpected OpenMP clause 'default' in directive '#pragma omp target'}}
  for (int i = 0; i < 200; i++) {
    ++x;
    ++y;
  }

int j = 0, i = 0, nn = 10;
#pragma omp target teams distribute simd default(shared) // expected-error {{unexpected OpenMP clause 'default' in directive '#pragma omp target teams distribute simd'}}
	for (j = 0; j < nn; j++ ) {
	  for (i = 0; i < nn; i++ ) {
		;
	  }
	}
#endif 
 
  return 0;
}
