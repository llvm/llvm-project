// RUN: %clang_cc1 -x c -verify %s

// Primary fixed point types
// Without `-ffixed-point`, these keywords are now treated as typedef'd types or identifiers.
_Accum accum;    // expected-error{{unknown type name '_Accum'}}
_Fract fract;    // expected-error{{unknown type name '_Fract'}}
_Sat _Accum sat_accum;    // expected-error{{unknown type name '_Sat'}}
                          // expected-error@-1{{expected ';' after top level declarator}}
signed _Accum signed_accum;    // expected-error{{expected ';' after top level declarator}}
signed _Fract signed_fract;    // expected-error{{expected ';' after top level declarator}}
signed _Sat _Accum signed_sat_accum;    // expected-error{{expected ';' after top level declarator}}

// Cannot use fixed point suffixes
int accum_int = 10k;     // expected-error{{invalid suffix 'k' on integer constant}}
int fract_int = 10r;     // expected-error{{invalid suffix 'r' on integer constant}}
float accum_flt = 10.0k; // expected-error{{invalid suffix 'k' on floating constant}}
float fract_flt = 10.0r; // expected-error{{invalid suffix 'r' on floating constant}}
