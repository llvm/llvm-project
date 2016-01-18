// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s

int foo();

template <class T>
T foo() {
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
    L1:
      foo();
    #pragma omp ordered
    {
      foo();
      goto L1; // expected-error {{use of undeclared label 'L1'}}
    }
  }
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
    foo();
    goto L2; // expected-error {{use of undeclared label 'L2'}}
    #pragma omp ordered
    {
      L2:
      foo();
    }
  }
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered threads threads // expected-error {{directive '#pragma omp ordered' cannot contain more than one 'threads' clause}}
    {
      foo();
    }
  }
  #pragma omp for ordered(1) // expected-note {{'ordered' clause with specified parameter}}
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered // expected-error {{'ordered' directive without any clauses cannot be closely nested inside ordered region with specified parameter}}
    {
      foo();
    }
  }
  #pragma omp for ordered(1) // expected-note {{'ordered' clause with specified parameter}}
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered threads // expected-error {{'ordered' directive with 'threads' clause cannot be closely nested inside ordered region with specified parameter}}
    {
      foo();
    }
  }
  #pragma omp ordered simd simd // expected-error {{directive '#pragma omp ordered' cannot contain more than one 'simd' clause}}
  {
    foo();
  }
  #pragma omp simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }
  #pragma omp simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered threads // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }
  #pragma omp for simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }
  #pragma omp for simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered threads // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }
  #pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }
  #pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered threads // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }

  return T();
}

int foo() {
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
    L1:
      foo();
    #pragma omp ordered
    {
      foo();
      goto L1; // expected-error {{use of undeclared label 'L1'}}
    }
  }
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
    foo();
    goto L2; // expected-error {{use of undeclared label 'L2'}}
    #pragma omp ordered
    {
      L2:
      foo();
    }
  }
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered threads threads // expected-error {{directive '#pragma omp ordered' cannot contain more than one 'threads' clause}}
    {
      foo();
    }
  }
  #pragma omp for ordered(1) // expected-note {{'ordered' clause with specified parameter}}
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered // expected-error {{'ordered' directive without any clauses cannot be closely nested inside ordered region with specified parameter}}
    {
      foo();
    }
  }
  #pragma omp for ordered(1) // expected-note {{'ordered' clause with specified parameter}}
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered threads // expected-error {{'ordered' directive with 'threads' clause cannot be closely nested inside ordered region with specified parameter}}
    {
      foo();
    }
  }
  #pragma omp ordered simd simd // expected-error {{directive '#pragma omp ordered' cannot contain more than one 'simd' clause}}
  {
    foo();
  }
  #pragma omp simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }
  #pragma omp simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered threads // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }
  #pragma omp for simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }
  #pragma omp for simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered threads // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }
  #pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }
  #pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered threads // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      foo();
    }
  }

  return foo<int>();
}
