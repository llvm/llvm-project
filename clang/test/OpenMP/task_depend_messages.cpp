// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-version=45 -fopenmp -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-version=50 -fopenmp -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp45 -fopenmp-version=45 -fopenmp-simd -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp50 -fopenmp-version=50 -fopenmp-simd -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

typedef void *omp_depend_t;

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

  #pragma omp task depend(in : arr[0])
  #pragma omp task depend // expected-error {{expected '(' after 'depend'}}
  #pragma omp task depend ( // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after dependency type - ignoring}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
  #pragma omp task depend () // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
  #pragma omp task depend (argc // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
  #pragma omp task depend (source : argc) // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
  #pragma omp task depend (source) // expected-error {{expected expression}} expected-warning {{missing ':' after dependency type - ignoring}} omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
  #pragma omp task depend (in : argc)) // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  #pragma omp task depend (out: ) // expected-error {{expected expression}}
  #pragma omp task depend (inout : foobool(argc)), depend (in, argc) // omp50-error {{expected addressable lvalue expression, array element, array section or array shaping expression}} omp45-error {{expected addressable lvalue expression, array element or array section}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected expression}}
  #pragma omp task depend (out :S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp task depend(in : argv[1][1] = '2')
  #pragma omp task depend (in : vec[1]) // omp50-error {{expected addressable lvalue expression, array element, array section or array shaping expression}} omp45-error {{expected addressable lvalue expression, array element or array section}}
  #pragma omp task depend (in : argv[0])
  #pragma omp task depend (in : ) // expected-error {{expected expression}}
  #pragma omp task depend (in : main)
  #pragma omp task depend(in : a[0]) // omp50-error {{expected addressable lvalue expression, array element, array section or array shaping expression}} omp45-error {{expected addressable lvalue expression, array element or array section}}
  #pragma omp task depend (in : vec[1:2]) // expected-error {{ value is not an array or pointer}}
  #pragma omp task depend (in : argv[ // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp task depend (in : argv[: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp task depend (in : argv[:] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task depend (in : argv[argc: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp task depend (in : argv[argc:argc] // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task depend (in : argv[0:-1]) // expected-error {{section length is evaluated to a negative value -1}}
  #pragma omp task depend (in : argv[-1:0]) // expected-error {{zero-length array section is not allowed in 'depend' clause}}
  #pragma omp task depend (in : argv[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma omp task depend (in : argv[3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  #pragma omp task depend(in:a[0:1]) // expected-error {{subscripted value is not an array or pointer}}
  #pragma omp task depend(in:argv[argv[:2]:1]) // expected-error {{OpenMP array section is not allowed here}}
  #pragma omp task depend(in:argv[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma omp task depend(in:env[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}}
  #pragma omp task depend(in : argv[ : argc][1 : argc - 1])
  #pragma omp task depend(in : arr[0])
  #pragma omp task depend(depobj:argc) // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp50-error {{expected lvalue expression of 'omp_depend_t' type, not 'int'}}
  #pragma omp task depend(depobj : argv[ : argc][1 : argc - 1]) // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp50-error {{expected lvalue expression of 'omp_depend_t' type, not '<OpenMP array section type>'}}
  #pragma omp task depend(depobj : arr[0]) // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}}
  #pragma omp task depend(in : ([ // expected-error {{expected variable name or 'this' in lambda capture list}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task depend(in : ([] // expected-error {{expected body of lambda expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task depend(in : ([]) // omp45-error {{expected body of lambda expression}} expected-error {{expected ')'}} expected-note {{to match this '('}} omp50-error 2 {{expected expression}}
  #pragma omp task depend(in : ([])a // omp45-error {{expected body of lambda expression}} expected-error {{expected ')'}} expected-note {{to match this '('}} omp50-error {{expected expression}}
  #pragma omp task depend(in : ([])a) // omp45-error {{expected body of lambda expression}} omp50-error {{expected expression}}
  #pragma omp task depend(in : ([a])a) // omp45-error {{expected body of lambda expression}} omp50-error {{expected expression with a pointer to a complete type as a base of an array shaping operation}}
  #pragma omp task depend(in : ([a])argc) // omp45-error {{expected body of lambda expression}} omp50-error {{expected expression with a pointer to a complete type as a base of an array shaping operation}}
  #pragma omp task depend(in : ([-1][0])argv) // omp45-error {{expected variable name or 'this' in lambda capture list}} omp45-error {{expected ')'}} omp45-note {{to match this '('}} omp50-error {{array shaping dimension is evaluated to a non-positive value -1}} omp50-error {{array shaping dimension is evaluated to a non-positive value 0}}
  #pragma omp task depend(iterator // expected-error {{expected ')'}} omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-note {{to match this '('}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} omp50-error {{expected '(' after 'iterator'}} omp50-error {{expected ','}}
  #pragma omp task depend(iterator():argc) // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp50-error {{expected ','}} omp50-error {{expected 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
  #pragma omp task depend(iterator(argc // expected-error {{expected ')'}} omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-note {{to match this '('}} omp50-error {{unknown type name 'argc'}} omp50-error {{expected ')'}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} omp50-error {{expected ','}} omp50-note {{to match this '('}}
  #pragma omp task depend(iterator(unsigned argc: // expected-error {{expected ')'}} omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-note {{to match this '('}} omp50-error {{expected '=' in iterator specifier}} omp50-error 2 {{expected expression}} omp50-error {{expected ',' or ')' after iterator specifier}} omp50-error {{expected ')'}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} omp50-error {{expected ','}} omp50-note {{to match this '('}}
  #pragma omp task depend(iterator(unsigned argc = // expected-error {{expected ')'}} omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-note {{to match this '('}} omp50-error 2 {{expected expression}} omp50-error {{expected ',' or ')' after iterator specifier}} omp50-error {{expected ')'}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} omp50-error {{expected ','}} omp50-note {{to match this '('}}
  #pragma omp task depend(iterator(vector argc = 0:2):argc) // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp50-error {{expected integral or pointer type as the iterator-type, not 'vector'}} omp50-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} omp50-error {{expected ','}}
  #pragma omp task depend(iterator(vector *argc = nullptr:nullptr+2:0), in:argc) // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp50-error {{invalid operands to binary expression ('nullptr_t' and 'int')}} omp50-error {{iterator step expression 0 evaluates to 0}}
  #pragma omp task depend(iterator(vector *argc = 0:vector():argc), in:argc) // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp50-error {{converting 'vector' to incompatible type 'vector *'}}
  foo();
#pragma omp task depend(iterator(unsigned argc = 0:10), in : argc) // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}}
  argc = 0;
#pragma omp task depend(iterator(i = 0:10, i = 0:10), in : argv[i]) // omp45-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp45-error {{use of undeclared identifier 'i'}} omp50-error {{redefinition of 'i'}} omp50-note {{previous definition is here}}
  i = 0; // expected-error {{use of undeclared identifier 'i'}}

  return 0;
}
