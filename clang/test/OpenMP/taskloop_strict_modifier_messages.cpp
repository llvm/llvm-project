// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp -fopenmp-version=51 -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -fopenmp-version=50 -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp -fopenmp-version=51 -fopenmp-simd -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -fopenmp-version=50 -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

template <class T, class S>
int tmain(T argc, S **argv) {
  T z;
  #pragma omp taskloop grainsize(strict 10) // omp51-error {{missing ':' after strict modifier}} omp50-error {{use of undeclared identifier 'strict'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp taskloop grainsize(aa 10) // omp51-error {{use of undeclared identifier 'aa'}} omp51-error {{expected ')'}} omp51-note {{to match this '('}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp taskloop grainsize(aa: 10) // omp51-error {{expected 'strict' in OpenMP clause 'grainsize'}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
     foo();

  return 0;
}

int main(int argc, char **argv) {
  int z;
  #pragma omp masked taskloop grainsize(strict 10) // omp51-error {{missing ':' after strict modifier}} omp50-error {{use of undeclared identifier 'strict'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp masked taskloop grainsize(aa 10) // omp51-error {{use of undeclared identifier 'aa'}} omp51-error {{expected ')'}} omp51-note {{to match this '('}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp masked taskloop grainsize(aa: 10) // omp51-error {{expected 'strict' in OpenMP clause 'grainsize'}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
     foo();
  
  #pragma omp masked taskloop simd grainsize(strict 10) // omp51-error {{missing ':' after strict modifier}} omp50-error {{use of undeclared identifier 'strict'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp masked taskloop simd grainsize(aa 10) // omp51-error {{use of undeclared identifier 'aa'}} omp51-error {{expected ')'}} omp51-note {{to match this '('}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp masked taskloop simd grainsize(aa: 10) // omp51-error {{expected 'strict' in OpenMP clause 'grainsize'}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
     foo();
  
  #pragma omp parallel masked taskloop grainsize(strict 10) // omp51-error {{missing ':' after strict modifier}} omp50-error {{use of undeclared identifier 'strict'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp parallel masked taskloop grainsize(aa 10) // omp51-error {{use of undeclared identifier 'aa'}} omp51-error {{expected ')'}} omp51-note {{to match this '('}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp parallel masked taskloop grainsize(aa: 10) // omp51-error {{expected 'strict' in OpenMP clause 'grainsize'}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
     foo();
  
  #pragma omp parallel masked taskloop simd grainsize(strict 10) // omp51-error {{missing ':' after strict modifier}} omp50-error {{use of undeclared identifier 'strict'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp parallel masked taskloop simd grainsize(aa 10) // omp51-error {{use of undeclared identifier 'aa'}} omp51-error {{expected ')'}} omp51-note {{to match this '('}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp parallel masked taskloop simd grainsize(aa: 10) // omp51-error {{expected 'strict' in OpenMP clause 'grainsize'}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
     foo();
  
  #pragma omp master taskloop grainsize(strict 10) // omp51-error {{missing ':' after strict modifier}} omp50-error {{use of undeclared identifier 'strict'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop grainsize(aa 10) // omp51-error {{use of undeclared identifier 'aa'}} omp51-error {{expected ')'}} omp51-note {{to match this '('}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop grainsize(aa: 10) // omp51-error {{expected 'strict' in OpenMP clause 'grainsize'}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
     foo();
  
  #pragma omp master taskloop simd grainsize(strict 10) // omp51-error {{missing ':' after strict modifier}} omp50-error {{use of undeclared identifier 'strict'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop simd grainsize(aa 10) // omp51-error {{use of undeclared identifier 'aa'}} omp51-error {{expected ')'}} omp51-note {{to match this '('}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp master taskloop simd grainsize(aa: 10) // omp51-error {{expected 'strict' in OpenMP clause 'grainsize'}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
     foo();
  
  #pragma omp parallel master taskloop grainsize(strict 10) // omp51-error {{missing ':' after strict modifier}} omp50-error {{use of undeclared identifier 'strict'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp parallel master taskloop grainsize(aa 10) // omp51-error {{use of undeclared identifier 'aa'}} omp51-error {{expected ')'}} omp51-note {{to match this '('}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp parallel master taskloop grainsize(aa: 10) // omp51-error {{expected 'strict' in OpenMP clause 'grainsize'}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
     foo();
  
  #pragma omp parallel master taskloop simd grainsize(strict 10) // omp51-error {{missing ':' after strict modifier}} omp50-error {{use of undeclared identifier 'strict'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp parallel master taskloop simd grainsize(aa 10) // omp51-error {{use of undeclared identifier 'aa'}} omp51-error {{expected ')'}} omp51-note {{to match this '('}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
    foo();
  #pragma omp parallel master taskloop simd grainsize(aa: 10) // omp51-error {{expected 'strict' in OpenMP clause 'grainsize'}} omp50-error {{use of undeclared identifier 'aa'}} omp50-error {{expected ')'}} omp50-note {{to match this '('}}
  for (int i = 0; i < 10; ++i)
     foo();

  return tmain(argc, argv);
}
