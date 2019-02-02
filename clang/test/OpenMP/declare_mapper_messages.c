// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s

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

#pragma omp declare mapper(++: struct vec v) map(v.len)                 // expected-error {{illegal identifier on 'omp declare mapper' directive}}
#pragma omp declare mapper(id1: struct vec v) map(v.len, temp)          // expected-error {{only variable v is allowed in map clauses of this 'omp declare mapper' directive}}
#pragma omp declare mapper(default : struct vec kk) map(kk.data[0:2])   // expected-note {{previous definition is here}}
#pragma omp declare mapper(struct vec v) map(v.len)                     // expected-error {{redefinition of user-defined mapper for type 'struct vec' with name 'default'}}
#pragma omp declare mapper(int v) map(v)                                // expected-error {{mapper type must be of struct, union or class type}}

int fun(int arg) {
#pragma omp declare mapper(id: struct vec v) map(v.len)
  {
#pragma omp declare mapper(id: struct vec v) map(v.len)                 // expected-note {{previous definition is here}}
#pragma omp declare mapper(id: struct vec v) map(v.len)                 // expected-error {{redefinition of user-defined mapper for type 'struct vec' with name 'id'}}
    {
#pragma omp declare mapper(id: struct vec v) map(v.len)
    }
  }
  return arg;
}
