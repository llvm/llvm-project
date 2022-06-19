// RUN: %clang_cc1 -verify -fopenmp-version=50 -fopenmp -ferror-limit 100 -o - -std=c++11 %s

// RUN: %clang_cc1 -verify -fopenmp-version=50 -fopenmp-simd -ferror-limit 100 -o - -std=c++11 %s

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

  #pragma omp task affinity(arr[0])
  #pragma omp task affinity // expected-error {{expected '(' after 'affinity'}}
  #pragma omp task affinity ( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  #pragma omp task affinity () // expected-error {{expected expression}}
  #pragma omp task affinity (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task affinity (argc)) // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  #pragma omp task affinity (foobool(argc)), affinity (in, argc) // expected-error {{expected addressable lvalue expression, array element, array section or array shaping expression}} expected-error {{use of undeclared identifier 'in'}}
  #pragma omp task affinity (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp task affinity(argv[1][1] = '2')
  #pragma omp task affinity (vec[1]) // expected-error {{expected addressable lvalue expression, array element, array section or array shaping expression}}
  #pragma omp task affinity (in: argv[0]) // expected-error {{use of undeclared identifier 'in'}}
  #pragma omp task affinity (main)
  #pragma omp task affinity(a[0]) // expected-error {{expected addressable lvalue expression, array element, array section or array shaping expression}}
  #pragma omp task affinity (vec[1:2]) // expected-error {{ value is not an array or pointer}}
  #pragma omp task affinity (argv[ // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp task affinity (argv[: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp task affinity (argv[:] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task affinity (argv[argc: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
  #pragma omp task affinity (argv[argc:argc] // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task affinity (argv[0:-1]) // expected-error {{section length is evaluated to a negative value -1}}
  #pragma omp task affinity (argv[-1:0])
  #pragma omp task affinity (argv[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma omp task affinity (argv[3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  #pragma omp task affinity(a[0:1]) // expected-error {{subscripted value is not an array or pointer}}
  #pragma omp task affinity(argv[argv[:2]:1]) // expected-error {{OpenMP array section is not allowed here}}
  #pragma omp task affinity(argv[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
  #pragma omp task affinity(env[0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}}
  #pragma omp task affinity(argv[ : argc][1 : argc - 1])
  #pragma omp task affinity(arr[0])
  #pragma omp task affinity(([ // expected-error {{expected variable name or 'this' in lambda capture list}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task affinity(([] // expected-error {{expected body of lambda expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task affinity(([]) // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error 2 {{expected expression}}
  #pragma omp task affinity(([])a // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  #pragma omp task affinity(([])a) // expected-error {{expected expression}}
  #pragma omp task affinity(([a])a) // expected-error {{expected expression with a pointer to a complete type as a base of an array shaping operation}}
  #pragma omp task affinity(([a])argc) // expected-error {{expected expression with a pointer to a complete type as a base of an array shaping operation}}
  #pragma omp task affinity(([-1][0])argv) // expected-error {{array shaping dimension is evaluated to a non-positive value -1}} expected-error {{array shaping dimension is evaluated to a non-positive value 0}}
  #pragma omp task affinity(iterator // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected '(' after 'iterator'}} expected-error {{expected expression}}
  #pragma omp task affinity(iterator():argc)
  #pragma omp task affinity(iterator(argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{unknown type name 'argc'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  #pragma omp task affinity(iterator(unsigned argc: // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected '=' in iterator specifier}} expected-error 2 {{expected expression}} expected-error {{expected ',' or ')' after iterator specifier}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  #pragma omp task affinity(iterator(unsigned argc = // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error 2 {{expected expression}} expected-error {{expected ',' or ')' after iterator specifier}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected expression}}
  #pragma omp task affinity(iterator(vector argc = 0:2):argc) // expected-error {{expected integral or pointer type as the iterator-type, not 'vector'}} expected-error {{expected expression}}
  #pragma omp task affinity(iterator(vector *argc = nullptr:nullptr+2:0):argc) // expected-error {{invalid operands to binary expression ('std::nullptr_t' and 'int')}} expected-error {{iterator step expression 0 evaluates to 0}} expected-error {{expected expression}}
  #pragma omp task affinity(iterator(vector *argc = 0:vector():argc):argc) // expected-error {{converting 'vector' to incompatible type 'vector *'}} expected-error {{expected expression}}
  foo();
#pragma omp task affinity(iterator(i = 0:10, i = 0:10) : argv[i]) // expected-error {{redefinition of 'i'}} expected-note {{previous definition is here}}
  i = 0; // expected-error {{use of undeclared identifier 'i'}}

  return 0;
}
