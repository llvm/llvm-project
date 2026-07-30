// RUN: %clang_cc1 -x c++ %s -verify -DWITHOUT_FIXED_POINT
// RUN: %clang_cc1 -x c++ %s -verify -ffixed-point

#ifdef WITHOUT_FIXED_POINT
_Accum accum;                           // expected-error{{unknown type name '_Accum'}}
_Fract fract;                           // expected-error{{unknown type name '_Fract'}}
_Sat _Accum sat_accum;                  // expected-error{{unknown type name '_Sat'}}
                                        // expected-error@-1{{expected ';' after top level declarator}}
#endif

int accum_int = 10k;     // expected-error{{invalid suffix 'k' on integer constant}}
int fract_int = 10r;     // expected-error{{invalid suffix 'r' on integer constant}}
#ifdef WITHOUT_FIXED_POINT
float accum_flt = 0.0k;  // expected-error{{invalid suffix 'k' on floating constant}}
float fract_flt = 0.0r;  // expected-error{{invalid suffix 'r' on floating constant}}
#endif

#ifndef WITHOUT_FIXED_POINT
const char *c = 10.0k;  // expected-error{{cannot initialize a variable of type 'const char *' with an rvalue of type '_Accum'}}
#endif
