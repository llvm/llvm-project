// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - -std=c++11 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}}

class vector {
  public:
    int operator[](int index) { return 0; }
};

template <class T, class S, class R>
int tmain(T argc, S **argv, R *env[]) {
  vector vec;
  typedef float V __attribute__((vector_size(16)));
  V a;
  char *arr;
  int i, z;

  #pragma omp depend target update to(z) // expected-error{{expected an OpenMP directive}}
  #pragma omp depend(out:argc) target update to(z) // expected-error{{expected an OpenMP directive}}
  #pragma omp target depend(in:argc) update to(z) // expected-error{{unexpected OpenMP clause 'update' in directive '#pragma omp target'}} expected-error{{unexpected OpenMP clause 'to' in directive '#pragma omp target'}}
  {}

  #pragma omp target update to(z) depend // expected-error {{expected '(' after 'depend'}}
  #pragma omp target update to(z) depend( // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma omp target update to(z) depend() // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma omp target update to(z) depend(argc // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(source : argc) // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}}
  #pragma omp target update to(z) depend(source) // expected-error {{expected expression}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma omp target update to(z) depend(in : argc)) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(z) depend(out: ) // expected-error {{expected expression}}
  #pragma omp target update to(z) depend(inout : foobool(argc)), depend (in, argc) // expected-error {{expected variable name, array element or array section}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected expression}}
  #pragma omp target update to(z) depend(out :S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp target update to(z) depend(in : argv[1][1] = '2') // expected-error {{expected variable name, array element or array section}}
  #pragma omp target update to(z) depend(in : vec[1]) // expected-error {{expected variable name, array element or array section}}
  #pragma omp target update to(z) depend(in : argv[0])
  #pragma omp target update to(z) depend(in : ) // expected-error {{expected expression}}
  #pragma omp target update to(z) depend(in : tmain) // expected-error {{expected variable name, array element or array section}}
  #pragma omp target update to(z) depend(in : a[0]) // expected-error{{expected variable name, array element or array section}}
  #pragma omp target update to(z) depend(in : vec[1:2]) // expected-error {{ value is not an array or pointer}}
  #pragma omp target update to(z) depend(in : argv[ // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(in : argv[: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(in : argv[:] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(in : argv[argc: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(in : argv[argc:argc] // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(in : argv[0:-1]) // expected-error {{section length is evaluated to a negative value -1}}
  #pragma omp target update to(z) depend(in : argv[-1:0]) // expected-error {{section lower bound is evaluated to a negative value -1}}
  #pragma omp target update to(z) depend(in : argv[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma omp target update to(z) depend(in : argv[3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  #pragma omp target update to(z) depend(in:a[0:1]) // expected-error {{subscripted value is not an array or pointer}}
  #pragma omp target update to(z) depend(in:argv[argv[:2]:1]) // expected-error {{OpenMP array section is not allowed here}}
  #pragma omp target update to(z) depend(in:argv[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma omp target update to(z) depend(in:env[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}}
  #pragma omp target update to(z) depend(in : argv[ : argc][1 : argc - 1])
  #pragma omp target update to(z) depend(in : arr[0])

  return 0;
}

int main(int argc, char **argv, char *env[]) {
  vector vec;
  typedef float V __attribute__((vector_size(16)));
  V a;
  auto arr = x; // expected-error {{use of undeclared identifier 'x'}}
  int z;

  #pragma omp depend target update to(z) // expected-error{{expected an OpenMP directive}}
  #pragma omp depend(out:argc) target update to(z) // expected-error{{expected an OpenMP directive}}
  #pragma omp target depend(in:argc) update to(z) // expected-error{{unexpected OpenMP clause 'update' in directive '#pragma omp target'}} expected-error{{unexpected OpenMP clause 'to' in directive '#pragma omp target'}}
  {}

  #pragma omp target update to(z) depend // expected-error {{expected '(' after 'depend'}}
  #pragma omp target update to(z) depend( // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma omp target update to(z) depend() // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma omp target update to(z) depend(argc // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(source : argc) // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}}
  #pragma omp target update to(z) depend(source) // expected-error {{expected expression}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma omp target update to(z) depend(in : argc)) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update to(z) depend(out: ) // expected-error {{expected expression}}
  #pragma omp target update to(z) depend(inout : foobool(argc)), depend (in, argc) // expected-error {{expected variable name, array element or array section}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected expression}}
  #pragma omp target update to(z) depend(out :S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp target update to(z) depend(in : argv[1][1] = '2') // expected-error {{expected variable name, array element or array section}}
  #pragma omp target update to(z) depend(in : vec[1]) // expected-error {{expected variable name, array element or array section}}
  #pragma omp target update to(z) depend(in : argv[0])
  #pragma omp target update to(z) depend(in : ) // expected-error {{expected expression}}
  #pragma omp target update to(z) depend(in : main) // expected-error {{expected variable name, array element or array section}}
  #pragma omp target update to(z) depend(in : a[0]) // expected-error{{expected variable name, array element or array section}}
  #pragma omp target update to(z) depend(in : vec[1:2]) // expected-error {{ value is not an array or pointer}}
  #pragma omp target update to(z) depend(in : argv[ // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(in : argv[: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(in : argv[:] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(in : argv[argc: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(in : argv[argc:argc] // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target update to(z) depend(in : argv[0:-1]) // expected-error {{section length is evaluated to a negative value -1}}
  #pragma omp target update to(z) depend(in : argv[-1:0]) // expected-error {{section lower bound is evaluated to a negative value -1}}
  #pragma omp target update to(z) depend(in : argv[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma omp target update to(z) depend(in : argv[3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  #pragma omp target update to(z) depend(in:a[0:1]) // expected-error {{subscripted value is not an array or pointer}}
  #pragma omp target update to(z) depend(in:argv[argv[:2]:1]) // expected-error {{OpenMP array section is not allowed here}}
  #pragma omp target update to(z) depend(in:argv[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma omp target update to(z) depend(in:env[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}}
  #pragma omp target update to(z) depend(in : argv[ : argc][1 : argc - 1])
  #pragma omp target update to(z) depend(in : arr[0])

  return tmain(argc, argv, env); // expected-note {{in instantiation of function template specialization 'tmain<int, char, char>' requested here}}
}
