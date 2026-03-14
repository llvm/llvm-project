// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

int temp; // expected-note 6 {{'temp' declared here}}

#pragma omp declare reduction                                              // expected-error {{expected '(' after 'declare reduction'}}
#pragma omp declare reduction {                                            // expected-error {{expected '(' after 'declare reduction'}}
#pragma omp declare reduction(                                             // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma omp declare reduction(#                                            // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma omp declare reduction(/                                            // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma omp declare reduction(+                                            // expected-error {{expected ':'}}
#pragma omp declare reduction(for                                          // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}}
#pragma omp declare reduction(if:                                          // expected-error {{expected identifier or one of the following operators: '+', '-', '*', '&', '|', '^', '&&', or '||'}} expected-error {{expected a type}}
#pragma omp declare reduction(oper:                                        // expected-error {{expected a type}}
#pragma omp declare reduction(oper;                                        // expected-error {{expected ':'}} expected-error {{expected a type}}
#pragma omp declare reduction(fun : int                                    // expected-error {{expected ':'}} expected-error {{expected expression}}
#pragma omp declare reduction(+ : const int:                               // expected-error {{reduction type cannot be qualified with 'const', 'volatile' or 'restrict'}}
#pragma omp declare reduction(- : volatile int:                            // expected-error {{reduction type cannot be qualified with 'const', 'volatile' or 'restrict'}}
#pragma omp declare reduction(* : int;                                     // expected-error {{expected ','}} expected-error {{expected a type}}
#pragma omp declare reduction(& : double char:                             // expected-error {{cannot combine with previous 'double' declaration specifier}} expected-error {{expected expression}}
#pragma omp declare reduction(^ : double, char, :                          // expected-error {{expected a type}} expected-error {{expected expression}}
#pragma omp declare reduction(&& : int, S:                                 // expected-error {{unknown type name 'S'}} expected-error {{expected expression}}
#pragma omp declare reduction(|| : int, double : temp += omp_in)           // expected-error 2 {{only 'omp_in' or 'omp_out' variables are allowed in combiner expression}}
#pragma omp declare reduction(| : char, float : omp_out += temp)           // expected-error 2 {{only 'omp_in' or 'omp_out' variables are allowed in combiner expression}}
#pragma omp declare reduction(fun : long : omp_out += omp_in) {            // expected-error {{expected 'initializer'}} expected-warning {{extra tokens at the end of '#pragma omp declare reduction' are ignored}}
#pragma omp declare reduction(fun : unsigned : omp_out += temp))           // expected-error {{expected 'initializer'}} expected-warning {{extra tokens at the end of '#pragma omp declare reduction' are ignored}} expected-error {{only 'omp_in' or 'omp_out' variables are allowed in combiner expression}}
#pragma omp declare reduction(fun : long(void) : omp_out += omp_in)        // expected-error {{reduction type cannot be a function type}}
#pragma omp declare reduction(fun : long[3] : omp_out += omp_in)           // expected-error {{reduction type cannot be an array type}}
#pragma omp declare reduction(fun23 : long, int, long : omp_out += omp_in) // expected-error {{redefinition of user-defined reduction for type 'long'}} expected-note {{previous definition is here}}

#pragma omp declare reduction(fun222 : long : omp_out += omp_in)
#pragma omp declare reduction(fun1 : long : omp_out += omp_in) initializer                 // expected-error {{expected '(' after 'initializer'}}
#pragma omp declare reduction(fun2 : long : omp_out += omp_in) initializer {               // expected-error {{expected '(' after 'initializer'}} expected-error {{expected expression}} expected-warning {{extra tokens at the end of '#pragma omp declare reduction' are ignored}}
#pragma omp declare reduction(fun3 : long : omp_out += omp_in) initializer[                // expected-error {{expected '(' after 'initializer'}} expected-error {{expected expression}} expected-warning {{extra tokens at the end of '#pragma omp declare reduction' are ignored}}
#pragma omp declare reduction(fun4 : long : omp_out += omp_in) initializer()               // expected-error {{expected expression}}
#pragma omp declare reduction(fun5 : long : omp_out += omp_in) initializer(temp)           // expected-error {{only 'omp_priv' or 'omp_orig' variables are allowed in initializer expression}}
#pragma omp declare reduction(fun6 : long : omp_out += omp_in) initializer(omp_orig        // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare reduction(fun7 : long : omp_out += omp_in) initializer(omp_priv 12)    // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare reduction(fun8 : long : omp_out += omp_in) initializer(omp_priv = 23)  // expected-note {{previous definition is here}}
#pragma omp declare reduction(fun8 : long : omp_out += omp_in) initializer(omp_priv = 23)) // expected-warning {{extra tokens at the end of '#pragma omp declare reduction' are ignored}} expected-error {{redefinition of user-defined reduction for type 'long'}}
#pragma omp declare reduction(fun9 : long : omp_out += omp_in) initializer(omp_priv = )    // expected-error {{expected expression}}

struct S {
  int s;
};
#pragma omp declare reduction(+: struct S: omp_out.s += omp_in.s) // initializer(omp_priv = { .s = 0 })
#pragma omp declare reduction(&: struct S: omp_out.s += omp_in.s) initializer(omp_priv = { .s = 0 })
#pragma omp declare reduction(|: struct S: omp_out.s += omp_in.s) initializer(omp_priv = { 0 })

int fun(int arg) {
  struct S s;
  s.s = 0;
#pragma omp parallel for reduction(+ : s)
  for (arg = 0; arg < 10; ++arg)
    s.s += arg;
#pragma omp declare reduction(red : int : omp_out++)
  {
#pragma omp declare reduction(red : int : omp_out++) // expected-note {{previous definition is here}}
#pragma omp declare reduction(red : int : omp_out++) // expected-error {{redefinition of user-defined reduction for type 'int'}}
    {
#pragma omp declare reduction(red : int : omp_out++)
    }
  }
  return arg;
}
