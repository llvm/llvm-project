// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp -fopenmp-version=51 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-simd -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp51 -fopenmp-simd -fopenmp-version=51 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

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

  #pragma omp taskwait depend // expected-error {{expected '(' after 'depend'}}
  #pragma omp taskwait depend ( // omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} omp51-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset', 'depobj' or 'inoutset' in OpenMP clause 'depend'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma omp taskwait depend () // omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} omp51-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset', 'depobj' or 'inoutset' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma omp taskwait depend (argc // omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} omp51-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset', 'depobj' or 'inoutset' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp taskwait depend (source : argc) // omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} omp51-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset', 'depobj' or 'inoutset' in OpenMP clause 'depend'}}
  #pragma omp taskwait depend (source) // expected-error {{expected expression}} expected-warning {{missing ':' after dependency type - ignoring}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} omp51-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset', 'depobj' or 'inoutset' in OpenMP clause 'depend'}}
  #pragma omp taskwait depend(mutexinoutset: argc) // expected-error{{'mutexinoutset' modifier not allowed in 'depend' clause on 'taskwait' directive}}
  #pragma omp taskwait depend (in : argc)) // expected-warning {{extra tokens at the end of '#pragma omp taskwait' are ignored}}
  #pragma omp taskwait depend (out: ) // expected-error {{expected expression}}
  #pragma omp taskwait depend (inout : foobool(argc)), depend (in, argc) // expected-error {{expected addressable lvalue expression, array element, array section or array shaping expression}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected expression}}

  #pragma omp taskwait depend (out :S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp taskwait depend(in : argv[1][1] = '2')
  #pragma omp taskwait depend (in : vec[1]) // expected-error {{expected addressable lvalue expression, array element, array section or array shaping expression}}
  #pragma omp taskwait depend (in : argv[0])
  #pragma omp taskwait depend (in : ) // expected-error {{expected expression}}
  #pragma omp taskwait depend (in : main)
  #pragma omp taskwait depend(in : a[0]) // expected-error{{expected addressable lvalue expression, array element, array section or array shaping expression}}
  #pragma omp taskwait depend (in : vec[1:2]) // expected-error {{ value is not an array or pointer}}
  #pragma omp taskwait depend (in : argv[ // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp taskwait depend (in : argv[: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp taskwait depend (in : argv[:] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp taskwait depend (in : argv[argc: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp taskwait depend (in : argv[argc:argc] // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp taskwait depend (in : argv[0:-1]) // expected-error {{section length is evaluated to a negative value -1}}
  #pragma omp taskwait depend (in : argv[-1:0]) // expected-error {{zero-length array section is not allowed in 'depend' clause}}
  #pragma omp taskwait depend (in : argv[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma omp taskwait depend (in : argv[3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  #pragma omp taskwait depend(in:a[0:1]) // expected-error {{subscripted value is not an array or pointer}}
  #pragma omp taskwait depend(in:argv[argv[:2]:1]) // expected-error {{OpenMP array section is not allowed here}}
  #pragma omp taskwait depend(in:argv[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma omp taskwait depend(in:env[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}}
  #pragma omp taskwait depend(in : argv[ : argc][1 : argc - 1])
  #pragma omp taskwait depend(in : arr[0])
  foo();

  return 0;
}
