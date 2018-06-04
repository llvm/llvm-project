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
                                      // expected-warning@-1{{type specifier missing, defaults to 'int'}}
long _Accum long_accum;               // expected-error{{compile with '-ffixed-point' to enable fixed point types}}
