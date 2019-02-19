// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++98 %s
// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++11 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++98 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++11 %s

int temp; // expected-note {{'temp' declared here}}

class vec {                                                             // expected-note {{definition of 'vec' is not complete until the closing '}'}}
private:
  int p;                                                                // expected-note {{declared private here}}
public:
  int len;
#pragma omp declare mapper(id: vec v) map(v.len)                        // expected-error {{member access into incomplete type 'vec'}}
  double *data;
};

#pragma omp declare mapper                                              // expected-error {{expected '(' after 'declare mapper'}}
#pragma omp declare mapper {                                            // expected-error {{expected '(' after 'declare mapper'}}
#pragma omp declare mapper(                                             // expected-error {{expected a type}} expected-error {{expected declarator on 'omp declare mapper' directive}}
#pragma omp declare mapper(#                                            // expected-error {{expected a type}} expected-error {{expected declarator on 'omp declare mapper' directive}}
#pragma omp declare mapper(v                                            // expected-error {{unknown type name 'v'}} expected-error {{expected declarator on 'omp declare mapper' directive}}
#pragma omp declare mapper(vec                                          // expected-error {{expected declarator on 'omp declare mapper' directive}}
#pragma omp declare mapper(S v                                          // expected-error {{unknown type name 'S'}}
#pragma omp declare mapper(vec v                                        // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare mapper(aa: vec v)                                   // expected-error {{expected at least one clause on '#pragma omp declare mapper' directive}}
#pragma omp declare mapper(bb: vec v) private(v)                        // expected-error {{expected at least one clause on '#pragma omp declare mapper' directive}} // expected-error {{unexpected OpenMP clause 'private' in directive '#pragma omp declare mapper'}}
#pragma omp declare mapper(cc: vec v) map(v) (                          // expected-warning {{extra tokens at the end of '#pragma omp declare mapper' are ignored}}

#pragma omp declare mapper(++: vec v) map(v.len)                        // expected-error {{illegal identifier on 'omp declare mapper' directive}}
#pragma omp declare mapper(id1: vec v) map(v.len, temp)                 // expected-error {{only variable v is allowed in map clauses of this 'omp declare mapper' directive}}
#pragma omp declare mapper(default : vec kk) map(kk.data[0:2])          // expected-note {{previous definition is here}}
#pragma omp declare mapper(vec v) map(v.len)                            // expected-error {{redefinition of user-defined mapper for type 'vec' with name 'default'}}
#pragma omp declare mapper(int v) map(v)                                // expected-error {{mapper type must be of struct, union or class type}}
#pragma omp declare mapper(id2: vec v) map(v.len, v.p)                  // expected-error {{'p' is a private member of 'vec'}}

namespace N1 {
template <class T>
class stack {                                                           // expected-note {{template is declared here}}
public:
  int len;
  T *data;
#pragma omp declare mapper(id: vec v) map(v.len)                        // expected-note {{previous definition is here}}
#pragma omp declare mapper(id: vec v) map(v.len)                        // expected-error {{redefinition of user-defined mapper for type 'vec' with name 'id'}}
};
};

#pragma omp declare mapper(default : N1::stack s) map(s.len)            // expected-error {{use of class template 'N1::stack' requires template arguments}}
#pragma omp declare mapper(id1: N1::stack<int> s) map(s.data)
#pragma omp declare mapper(default : S<int> s) map(s.len)               // expected-error {{no template named 'S'}}

template <class T>
T foo(T a) {
#pragma omp declare mapper(id: vec v) map(v.len)                        // expected-note {{previous definition is here}}
#pragma omp declare mapper(id: vec v) map(v.len)                        // expected-error {{redefinition of user-defined mapper for type 'vec' with name 'id'}}
}

int fun(int arg) {
#pragma omp declare mapper(id: vec v) map(v.len)
  {
#pragma omp declare mapper(id: vec v) map(v.len)                        // expected-note {{previous definition is here}}
    {
#pragma omp declare mapper(id: vec v) map(v.len)
      vec vv, v1;
#pragma omp target map(mapper)                                          // expected-error {{use of undeclared identifier 'mapper'}}
      {}
#pragma omp target map(mapper:vv)                                       // expected-error {{expected '(' after 'mapper'}}
      {}
#pragma omp target map(mapper( :vv)                                     // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
      {}
#pragma omp target map(mapper(aa :vv)                                   // expected-error {{use of undeclared identifier 'aa'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
      {}
#pragma omp target map(mapper(ab) :vv)                                  // expected-error {{missing map type}} expected-error {{cannot find a valid user-defined mapper for type 'vec' with name 'ab'}}
      {}
#pragma omp target map(mapper(N2::) :vv)                                // expected-error {{use of undeclared identifier 'N2'}} expected-error {{illegal identifier on 'omp declare mapper' directive}}
      {}
#pragma omp target map(mapper(N1::) :vv)                                // expected-error {{illegal identifier on 'omp declare mapper' directive}}
      {}
#pragma omp target map(mapper(aa) :vv)                                  // expected-error {{missing map type}}
      {}
#pragma omp target map(mapper(N1::aa) alloc:vv)                         // expected-error {{cannot find a valid user-defined mapper for type 'vec' with name 'aa'}}
      {}
#pragma omp target map(mapper(aa) to:vv) map(close mapper(aa) from:v1)
      {}
#pragma omp target map(mapper(N1::stack<int>::id) to:vv)
      {}
    }
#pragma omp declare mapper(id: vec v) map(v.len)                        // expected-error {{redefinition of user-defined mapper for type 'vec' with name 'id'}}
  }
  return arg;
}
