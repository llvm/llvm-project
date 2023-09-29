// RUN: %clang_cc1 -x c++ %s -verify -DWITHOUT_FIXED_POINT
// RUN: %clang_cc1 -x c++ %s -verify -ffixed-point

#ifdef WITHOUT_FIXED_POINT
_Accum accum;                           // expected-error{{compile with '-ffixed-point' to enable fixed point types}}
                                        // expected-error@-1{{a type specifier is required for all declarations}}
_Fract fract;                           // expected-error{{compile with '-ffixed-point' to enable fixed point types}}
                                        // expected-error@-1{{a type specifier is required for all declarations}}
_Sat _Accum sat_accum;                  // expected-error 2{{compile with '-ffixed-point' to enable fixed point types}}
                                        // expected-error@-1{{a type specifier is required for all declarations}}
#endif

int accum_int = 10k;     // expected-error{{invalid suffix 'k' on integer constant}}
int fract_int = 10r;     // expected-error{{invalid suffix 'r' on integer constant}}
#ifdef WITHOUT_FIXED_POINT
float accum_flt = 0.0k;  // expected-error{{invalid suffix 'k' on floating constant}}
float fract_flt = 0.0r;  // expected-error{{invalid suffix 'r' on floating constant}}
#endif
