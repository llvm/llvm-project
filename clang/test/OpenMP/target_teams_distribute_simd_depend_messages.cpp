// RUN: %clang_cc1 -verify=expected,omp4 -fopenmp -fopenmp-version=45 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp5 -fopenmp -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp4 -fopenmp-simd -fopenmp-version=45 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp5 -fopenmp-simd -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

class vector {
  public:
    int operator[](int index) { return 0; }
};

int main(int argc, char **argv, char *env[]) {
  vector vec;
  typedef float V __attribute__((vector_size(16)));
  V a;
  auto arr = x; // expected-error {{use of undeclared identifier 'x'}}
  int i;

#pragma omp target teams distribute simd depend // expected-error {{expected '(' after 'depend'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend( // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after dependency type - ignoring}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend() // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(argc // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(source : argc) // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target teams distribute simd depend(source) // expected-error {{expected expression}} expected-warning {{missing ':' after dependency type - ignoring}} omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (out: ) // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target teams distribute simd depend(inout : foobool(argc)), depend(in, argc) // omp4-error {{expected addressable lvalue expression, array element or array section}} omp5-error {{expected addressable lvalue expression, array element, array section or array shaping expression of non 'omp_depend_t' type}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (out :S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i)
    foo();
#pragma omp target teams distribute simd depend(in : argv[1][1] = '2')
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(in : vec[1]) // omp4-error {{expected addressable lvalue expression, array element or array section}} omp5-error {{expected addressable lvalue expression, array element, array section or array shaping expression of non 'omp_depend_t' type}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : argv[0])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(in:) // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : main)
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(in : a[0]) // omp4-error {{expected addressable lvalue expression, array element or array section}} omp5-error {{expected addressable lvalue expression, array element, array section or array shaping expression of non 'omp_depend_t' type}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : vec[1:2]) // expected-error {{ value is not an array or pointer}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : argv[ // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : argv[: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : argv[:] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : argv[argc: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : argv[argc:argc] // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : argv[0:-1]) // expected-error {{section length is evaluated to a negative value -1}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : argv[-1:0]) // expected-error {{zero-length array section is not allowed in 'depend' clause}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend (in : argv[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(in : argv [3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(in:a[0:1]) // expected-error {{subscripted value is not an array or pointer}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(in:argv[argv[:2]:1]) // expected-error {{OpenMP array section is not allowed here}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(in:argv[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(in:env[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(in : argv[ : argc][1 : argc - 1])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute simd depend(in : arr[0])
  for (i = 0; i < argc; ++i) foo();

  return 0;
}
