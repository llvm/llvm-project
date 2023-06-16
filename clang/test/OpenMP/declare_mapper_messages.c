// RUN: %clang_cc1 -verify=omp50,expected -fopenmp -fopenmp-version=50 -ferror-limit 100 -DOMP50 %s
// RUN: %clang_cc1 -verify=omp51,expected -fopenmp -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -ferror-limit 100 -DOMP52 %s

// RUN: %clang_cc1 -verify=omp50,expected -fopenmp-simd -fopenmp-version=50 -ferror-limit 100 -DOMP50 %s
// RUN: %clang_cc1 -verify=omp51-simd,expected -fopenmp-simd -ferror-limit 100 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -ferror-limit 100 -DOMP52 %s

int temp; // expected-note {{'temp' declared here}}

struct vec {                                                            // expected-note {{definition of 'struct vec' is not complete until the closing '}'}}
  int len;
#pragma omp declare mapper(id: struct vec v) map(v.len)                 // expected-error {{incomplete definition of type 'struct vec'}}
  double *data;
};

#pragma omp declare mapper                                              // expected-error {{expected '(' after 'declare mapper'}}
#pragma omp declare mapper {                                            // expected-error {{expected '(' after 'declare mapper'}}
#pragma omp declare mapper(                                             // expected-error {{expected a type}} expected-error {{expected declarator on 'omp declare mapper' directive}}
#pragma omp declare mapper(#                                            // expected-error {{expected a type}} expected-error {{expected declarator on 'omp declare mapper' directive}}
#pragma omp declare mapper(struct v                                     // expected-error {{expected declarator on 'omp declare mapper' directive}}
#pragma omp declare mapper(struct vec                                   // expected-error {{expected declarator on 'omp declare mapper' directive}}
#pragma omp declare mapper(S v                                          // expected-error {{unknown type name 'S'}}
#pragma omp declare mapper(struct vec v                                 // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare mapper(aa:struct vec v)                             // expected-error {{expected at least one clause on '#pragma omp declare mapper' directive}}
#pragma omp declare mapper(bb:struct vec v) private(v)                  // expected-error {{expected at least one clause on '#pragma omp declare mapper' directive}} // expected-error {{unexpected OpenMP clause 'private' in directive '#pragma omp declare mapper'}}
#pragma omp declare mapper(cc:struct vec v) map(v) (                    // expected-warning {{extra tokens at the end of '#pragma omp declare mapper' are ignored}}

#pragma omp declare mapper(++: struct vec v) map(v.len)                 // expected-error {{illegal OpenMP user-defined mapper identifier}}
#pragma omp declare mapper(id1: struct vec v) map(v.len, temp)          // expected-error {{only variable 'v' is allowed in map clauses of this 'omp declare mapper' directive}}
#pragma omp declare mapper(default : struct vec kk) map(kk.data[0:2])   // expected-note {{previous definition is here}}
#pragma omp declare mapper(struct vec v) map(v.len)                     // expected-error {{redefinition of user-defined mapper for type 'struct vec' with name 'default'}}
#pragma omp declare mapper(int v) map(v)                                // expected-error {{mapper type must be of struct, union or class type}}

#ifndef OMP52
// omp51-simd-error@+6 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'ompx_hold'}} 
// omp50-error@+5 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'ompx_hold'}} 
// omp51-error@+4 {{incorrect map type modifier, expected one of: 'always', 'close', 'mapper', 'present', 'ompx_hold'}} 
// expected-error@+3 {{only variable 'vvec' is allowed in map clauses of this 'omp declare mapper' directive}} 
// expected-error@+2 {{expected at least one clause on '#pragma omp declare mapper' directive}}
// expected-note@+1 {{'it' declared here}}
#pragma omp declare mapper(id2: struct vec vvec) map(iterator(it=0:vvec.len:2), tofrom:vvec.data[it]) 

#else
#pragma omp declare mapper(id2: struct vec vvec) map(iterator(it=0:vvec.len:2), tofrom:vvec.data[it])

int var; // expected-note {{'var' declared here}}
// expected-error@+1 {{only variable 'vvec' is allowed in map clauses of this 'omp declare mapper' directive}}
#pragma omp declare mapper(id3: struct vec vvec) map(iterator(it=0:vvec.len:2), tofrom:vvec.data[var])

#endif // OMP52

int fun(int arg) {
#pragma omp declare mapper(id: struct vec v) map(v.len)
  {
#pragma omp declare mapper(id: struct vec v) map(v.len)                 // expected-note {{previous definition is here}}
#pragma omp declare mapper(id: struct vec v) map(v.len)                 // expected-error {{redefinition of user-defined mapper for type 'struct vec' with name 'id'}}
    {
#pragma omp declare mapper(id: struct vec v) map(v.len) allocate(v)   // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp declare mapper'}}
      struct vec vv, v1;
      struct vec arr[10];
      double d;
#pragma omp target map(mapper)                                          // expected-error {{use of undeclared identifier 'mapper'}}
      {}
#pragma omp target map(mapper:vv)                                       // expected-error {{expected '(' after 'mapper'}}
      {}
#pragma omp target map(mapper( :vv)                                     // expected-error {{expected expression}} expected-error {{expected ')'}} expected-error {{call to undeclared function 'mapper'}} expected-note {{to match this '('}}
      {}
#pragma omp target map(mapper(aa :vv)                                   // expected-error {{use of undeclared identifier 'aa'}} expected-error {{expected ')'}} expected-error {{call to undeclared function 'mapper'}} expected-note {{to match this '('}}
      {}
#pragma omp target map(mapper(ab) :vv)                                  // expected-error {{missing map type}} expected-error {{cannot find a valid user-defined mapper for type 'struct vec' with name 'ab'}}
      {}
#pragma omp target map(mapper(ab) :arr[0:2])                            // expected-error {{missing map type}} expected-error {{cannot find a valid user-defined mapper for type 'struct vec' with name 'ab'}}
      {}
#pragma omp target map(mapper(aa) :vv)                                  // expected-error {{missing map type}}
      {}
#pragma omp target map(mapper(aa) to:d)                                 // expected-error {{mapper type must be of struct, union or class type}}
      {}
#pragma omp target map(mapper(aa) to:vv) map(close mapper(aa) from:v1) map(mapper(aa) to:arr[0])
      {}

#pragma omp target update to(mapper)                                    // expected-error {{expected '(' after 'mapper'}} expected-error {{expected expression}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(mapper()                                   // expected-error {{illegal OpenMP user-defined mapper identifier}} expected-error {{expected expression}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(mapper:vv)                                 // expected-error {{expected '(' after 'mapper'}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(mapper(:vv)                                // expected-error {{illegal OpenMP user-defined mapper identifier}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(mapper(aa :vv)                             // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(mapper(ab):vv)                             // expected-error {{cannot find a valid user-defined mapper for type 'struct vec' with name 'ab'}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(mapper(ab):arr[0:2])                       // expected-error {{cannot find a valid user-defined mapper for type 'struct vec' with name 'ab'}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#ifdef OMP50
#pragma omp target update to(mapper(aa) a:vv)                           // expected-warning {{missing ':' after ) - ignoring}}
#else
#pragma omp target update to(mapper(aa) a:vv)                           // expected-warning {{missing ':' after motion modifier - ignoring}}
#endif
#pragma omp target update to(mapper(aa):d)                              // expected-error {{mapper type must be of struct, union or class type}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(mapper(aa):vv) to(mapper(aa):arr[0])

#pragma omp target update from(mapper)                                  // expected-error {{expected '(' after 'mapper'}} expected-error {{expected expression}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(mapper()                                 // expected-error {{illegal OpenMP user-defined mapper identifier}} expected-error {{expected expression}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(mapper:vv)                               // expected-error {{expected '(' after 'mapper'}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(mapper(:vv)                              // expected-error {{illegal OpenMP user-defined mapper identifier}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(mapper(aa :vv)                           // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(mapper(ab):vv)                           // expected-error {{cannot find a valid user-defined mapper for type 'struct vec' with name 'ab'}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(mapper(ab):arr[0:2])                     // expected-error {{cannot find a valid user-defined mapper for type 'struct vec' with name 'ab'}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#ifdef OMP50
#pragma omp target update from(mapper(aa) a:vv)                         // expected-warning {{missing ':' after ) - ignoring}}
#else
#pragma omp target update from(mapper(aa) a:vv)                         // expected-warning {{missing ':' after motion modifier - ignoring}}
#endif
#pragma omp target update from(mapper(aa):d)                            // expected-error {{mapper type must be of struct, union or class type}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update from(mapper(aa):vv) from(mapper(aa):arr[0])
    }
  }
  return arg;
}
