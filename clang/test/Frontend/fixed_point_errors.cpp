// RUN: %clang_cc1 -x c++ %s -verify
// RUN: %clang_cc1 -x c++ -ffixed-point %s -verify

// Name namgling is not provided for fixed point types in c++

_Accum accum;                           // expected-error{{unknown type name '_Accum'}}
_Fract fract;                           // expected-error{{unknown type name '_Fract'}}
_Sat _Accum sat_accum;                  // expected-error{{unknown type name '_Sat'}}
                                        // expected-error@-1{{expected ';' after top level declarator}}
