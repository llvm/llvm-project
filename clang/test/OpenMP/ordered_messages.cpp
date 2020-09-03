// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++98 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++11 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++98 -o - %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++11 -o - %s -Wuninitialized

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered
    argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
  }
}

int foo();

template <class T>
T foo() {
 T k;
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
  #pragma omp for ordered
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered // expected-note {{previous 'ordered' directive used here}}
    {
      foo();
    }
    #pragma omp ordered // expected-error {{exactly one 'ordered' directive must appear in the loop body of an enclosing directive}}
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
  #pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp ordered simd // expected-note {{previous 'ordered' directive used here}}
    {
      foo();
    }
#pragma omp ordered simd // expected-error {{exactly one 'ordered' directive must appear in the loop body of an enclosing directive}}
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
    #pragma omp ordered depend(source) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  }
#pragma omp parallel for ordered
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered depend(source) // expected-error {{'ordered' directive with 'depend' clause cannot be closely nested inside ordered region without specified parameter}}
    #pragma omp ordered depend(sink : i) // expected-error {{'ordered' directive with 'depend' clause cannot be closely nested inside ordered region without specified parameter}}
  }
#pragma omp parallel for ordered(2) // expected-note 3 {{'ordered' clause with specified parameter}}
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
#pragma omp ordered depend // expected-error {{expected '(' after 'depend'}} expected-error {{'ordered' directive without any clauses cannot be closely nested inside ordered region with specified parameter}}
#pragma omp ordered depend( // expected-error {{expected ')'}} expected-error {{expected 'source' or 'sink' in OpenMP clause 'depend'}} expected-error {{'ordered' directive without any clauses cannot be closely nested inside ordered region with specified parameter}} expected-warning {{missing ':' or ')' after dependency type - ignoring}} expected-note {{to match this '('}}
#pragma omp ordered depend(source // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp ordered depend(sink // expected-error {{expected expression}} expected-warning {{missing ':' or ')' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected 'i' loop iteration variable}}
#pragma omp ordered depend(sink : // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}} expected-error {{expected 'i' loop iteration variable}}
#pragma omp ordered depend(sink : i // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected 'j' loop iteration variable}}
#pragma omp ordered depend(sink : i) // expected-error {{expected 'j' loop iteration variable}}
#pragma omp ordered depend(source)
                           if (i == j)
#pragma omp ordered depend(source) // expected-error {{'#pragma omp ordered' with 'depend' clause cannot be an immediate substatement}}
                             ;
                           if (i == j)
#pragma omp ordered depend(sink : i, j) // expected-error {{'#pragma omp ordered' with 'depend' clause cannot be an immediate substatement}}
                             ;
#pragma omp ordered depend(source) threads // expected-error {{'depend' clauses cannot be mixed with 'threads' clause}}
#pragma omp ordered simd depend(source) // expected-error {{'depend' clauses cannot be mixed with 'simd' clause}}
#pragma omp ordered depend(source) depend(source) // expected-error {{directive '#pragma omp ordered' cannot contain more than one 'depend' clause with 'source' dependence}}
#pragma omp ordered depend(in : i) // expected-error {{expected 'source' or 'sink' in OpenMP clause 'depend'}} expected-error {{'ordered' directive without any clauses cannot be closely nested inside ordered region with specified parameter}}
#pragma omp ordered depend(sink : i, j)
#pragma omp ordered depend(sink : j, i) // expected-error {{expected 'i' loop iteration variable}} expected-error {{expected 'j' loop iteration variable}}
#pragma omp ordered depend(sink : i, j, k) // expected-error {{unexpected expression: number of expressions is larger than the number of associated loops}}
#pragma omp ordered depend(sink : i+foo(), j/4) // expected-error {{expression is not an integral constant expression}} expected-error {{expected '+' or '-' operation}}
#if __cplusplus >= 201103L
// expected-note@-2 {{non-constexpr function 'foo' cannot be used in a constant expression}}
#endif
#pragma omp ordered depend(sink : i*0, j-4)// expected-error {{expected '+' or '-' operation}}
#pragma omp ordered depend(sink : i-0, j+sizeof(T)) depend(sink : i-0, j+sizeof(T))
#pragma omp ordered depend(sink : i-0, j+sizeof(T)) depend(source) // expected-error {{'depend(source)' clause cannot be mixed with 'depend(sink:vec)' clauses}}
#pragma omp ordered depend(source) depend(sink : i-0, j+sizeof(T)) // expected-error {{'depend(sink:vec)' clauses cannot be mixed with 'depend(source)' clause}}
    }
  }
#pragma omp ordered depend(source) // expected-error {{'ordered' directive with 'depend' clause cannot be closely nested inside ordered region without specified parameter}}
#pragma omp ordered depend(sink:k) // expected-error {{'ordered' directive with 'depend' clause cannot be closely nested inside ordered region without specified parameter}}
  return T();
}

int foo() {
#if __cplusplus >= 201103L
// expected-note@-2 2 {{declared here}}
#endif
int k;
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
    #pragma omp ordered depend(source) // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  }
#pragma omp parallel for ordered
  for (int i = 0; i < 10; ++i) {
    #pragma omp ordered depend(source) // expected-error {{'ordered' directive with 'depend' clause cannot be closely nested inside ordered region without specified parameter}}
    #pragma omp ordered depend(sink : i) // expected-error {{'ordered' directive with 'depend' clause cannot be closely nested inside ordered region without specified parameter}}
  }
#pragma omp parallel for ordered(2) // expected-note 3 {{'ordered' clause with specified parameter}}
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
#pragma omp ordered depend // expected-error {{expected '(' after 'depend'}} expected-error {{'ordered' directive without any clauses cannot be closely nested inside ordered region with specified parameter}}
#pragma omp ordered depend( // expected-error {{expected ')'}} expected-error {{expected 'source' or 'sink' in OpenMP clause 'depend'}} expected-error {{'ordered' directive without any clauses cannot be closely nested inside ordered region with specified parameter}} expected-warning {{missing ':' or ')' after dependency type - ignoring}} expected-note {{to match this '('}}
#pragma omp ordered depend(source // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp ordered depend(sink // expected-error {{expected expression}} expected-warning {{missing ':' or ')' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected 'i' loop iteration variable}}
#pragma omp ordered depend(sink : // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}} expected-error {{expected 'i' loop iteration variable}}
#pragma omp ordered depend(sink : i // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected 'j' loop iteration variable}}
#pragma omp ordered depend(sink : i) // expected-error {{expected 'j' loop iteration variable}}
#pragma omp ordered depend(source)
                           if (i == j)
#pragma omp ordered depend(source) // expected-error {{'#pragma omp ordered' with 'depend' clause cannot be an immediate substatement}}
                             ;
                           if (i == j)
#pragma omp ordered depend(sink : i, j) // expected-error {{'#pragma omp ordered' with 'depend' clause cannot be an immediate substatement}}
                             ;
#pragma omp ordered depend(source) threads // expected-error {{'depend' clauses cannot be mixed with 'threads' clause}}
#pragma omp ordered simd depend(source) // expected-error {{'depend' clauses cannot be mixed with 'simd' clause}}
#pragma omp ordered depend(source) depend(source) // expected-error {{directive '#pragma omp ordered' cannot contain more than one 'depend' clause with 'source' dependence}}
#pragma omp ordered depend(in : i) // expected-error {{expected 'source' or 'sink' in OpenMP clause 'depend'}} expected-error {{'ordered' directive without any clauses cannot be closely nested inside ordered region with specified parameter}}
#pragma omp ordered depend(sink : i, j) allocate(i) // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp ordered'}}
#pragma omp ordered depend(sink : j, i) // expected-error {{expected 'i' loop iteration variable}} expected-error {{expected 'j' loop iteration variable}}
#pragma omp ordered depend(sink : i, j, k) // expected-error {{unexpected expression: number of expressions is larger than the number of associated loops}}
#pragma omp ordered depend(sink : i+foo(), j/4) // expected-error {{expression is not an integral constant expression}} expected-error {{expected '+' or '-' operation}}
#if __cplusplus >= 201103L
// expected-note@-2 {{non-constexpr function 'foo' cannot be used in a constant expression}}
#endif
#pragma omp ordered depend(sink : i*0, j-4)// expected-error {{expected '+' or '-' operation}}
#pragma omp ordered depend(sink : i-0, j+sizeof(int)) depend(sink : i-0, j+sizeof(int))
#pragma omp ordered depend(sink : i-0, j+sizeof(int)) depend(source) // expected-error {{'depend(source)' clause cannot be mixed with 'depend(sink:vec)' clauses}}
#pragma omp ordered depend(source) depend(sink : i-0, j+sizeof(int)) // expected-error {{'depend(sink:vec)' clauses cannot be mixed with 'depend(source)' clause}}
    }
  }

#pragma omp for ordered(2) // expected-note {{as specified in 'ordered' clause}}
  for (int i = 0; i < 10; ++i) { // expected-error {{expected 2 for loops after '#pragma omp for', but found only 1}}
#pragma omp ordered depend(sink : i)
    int j;
#pragma omp ordered depend(sink : i, j) // expected-error {{expected loop iteration variable}}
    foo();
  }

  return foo<int>(); // expected-note {{in instantiation of function template specialization 'foo<int>' requested here}}
}
