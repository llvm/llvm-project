// RUN: %clang_cc1 -x c++ %s -verify
// RUN: %clang_cc1 -x c++ -ffixed-point %s -verify

// Name namgling is not provided for fixed point types in c++

_Accum accum;                           // expected-error{{unknown type name '_Accum'}}
_Fract fract;                           // expected-error{{unknown type name '_Fract'}}
_Sat _Accum sat_accum;                  // expected-error{{unknown type name '_Sat'}}
                                        // expected-error@-1{{expected ';' after top level declarator}}

int accum_int = 10k;     // expected-error{{invalid suffix 'k' on integer constant}}
int fract_int = 10r;     // expected-error{{invalid suffix 'r' on integer constant}}
float accum_flt = 10.0k; // expected-error{{invalid suffix 'k' on floating constant}}
float fract_flt = 10.0r; // expected-error{{invalid suffix 'r' on floating constant}}
