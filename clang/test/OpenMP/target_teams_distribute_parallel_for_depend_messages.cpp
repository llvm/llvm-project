// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - -std=c++11 %s

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

#pragma omp target teams distribute parallel for depend // expected-error {{expected '(' after 'depend'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend ( // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after dependency type - ignoring}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend () // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (argc // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (source : argc) // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (source) // expected-error {{expected expression}} expected-warning {{missing ':' after dependency type - ignoring}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute parallel for' are ignored}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (out: ) // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (inout : foobool(argc)), depend (in, argc) // expected-error {{expected variable name, array element or array section}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (out :S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend(in : argv[1][1] = '2') // expected-error {{expected variable name, array element or array section}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : vec[1]) // expected-error {{expected variable name, array element or array section}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : argv[0])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : ) // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : main) // expected-error {{expected variable name, array element or array section}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend(in : a[0]) // expected-error{{expected variable name, array element or array section}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : vec[1:2]) // expected-error {{ value is not an array or pointer}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : argv[ // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : argv[: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : argv[:] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : argv[argc: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : argv[argc:argc] // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : argv[0:-1]) // expected-error {{section length is evaluated to a negative value -1}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : argv[-1:0])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : argv[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend (in : argv[3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend(in:a[0:1]) // expected-error {{subscripted value is not an array or pointer}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend(in:argv[argv[:2]:1]) // expected-error {{OpenMP array section is not allowed here}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend(in:argv[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend(in:env[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}}
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend(in : argv[ : argc][1 : argc - 1])
  for (i = 0; i < argc; ++i) foo();
#pragma omp target teams distribute parallel for depend(in : arr[0])
  for (i = 0; i < argc; ++i) foo();

  return 0;
}
