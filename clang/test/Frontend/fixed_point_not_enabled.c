// RUN: %clang_cc1 -x c -verify %s

// Primary fixed point types
signed short _Accum s_short_accum;    // expected-error{{compile with '-ffixed-point' to enable fixed point types}}
signed _Accum s_accum;                // expected-error{{compile with '-ffixed-point' to enable fixed point types}}
signed long _Accum s_long_accum;      // expected-error{{compile with '-ffixed-point' to enable fixed point types}}
unsigned short _Accum u_short_accum;  // expected-error{{compile with '-ffixed-point' to enable fixed point types}}
unsigned _Accum u_accum;              // expected-error{{compile with '-ffixed-point' to enable fixed point types}}
unsigned long _Accum u_long_accum;    // expected-error{{compile with '-ffixed-point' to enable fixed point types}}

// Aliased fixed point types
short _Accum short_accum;             // expected-error{{compile with '-ffixed-point' to enable fixed point types}}
_Accum accum;                         // expected-error{{compile with '-ffixed-point' to enable fixed point types}}
                                      // expected-error@-1{{type specifier missing, defaults to 'int'}}
long _Accum long_accum;               // expected-error{{compile with '-ffixed-point' to enable fixed point types}}

// Cannot use fixed point suffixes
int accum_int = 10k;     // expected-error{{invalid suffix 'k' on integer constant}}
int fract_int = 10r;     // expected-error{{invalid suffix 'r' on integer constant}}
float accum_flt = 10.0k; // expected-error{{invalid suffix 'k' on floating constant}}
float fract_flt = 10.0r; // expected-error{{invalid suffix 'r' on floating constant}}
