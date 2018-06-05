// RUN: %clang_cc1 -verify -ffixed-point %s

/* We do not yet support long long. No recommended bit widths are given for this
 * size. */

long long _Accum longlong_accum;              // expected-error{{'long long _Accum' is invalid}}
unsigned long long _Accum u_longlong_accum;   // expected-error{{'long long _Accum' is invalid}}

/* Although _Complex types work with floating point numbers, the extension
 * provides no info for complex fixed point types. */

_Complex signed short _Accum cmplx_s_short_accum;   // expected-error{{'_Complex _Accum' is invalid}}
_Complex signed _Accum cmplx_s_accum;               // expected-error{{'_Complex _Accum' is invalid}}
_Complex signed long _Accum cmplx_s_long_accum;     // expected-error{{'_Complex _Accum' is invalid}}
_Complex unsigned short _Accum cmplx_u_short_accum; // expected-error{{'_Complex _Accum' is invalid}}
_Complex unsigned _Accum cmplx_u_accum;             // expected-error{{'_Complex _Accum' is invalid}}
_Complex unsigned long _Accum cmplx_u_long_accum;   // expected-error{{'_Complex _Accum' is invalid}}
_Complex short _Accum cmplx_s_short_accum;          // expected-error{{'_Complex _Accum' is invalid}}
_Complex _Accum cmplx_s_accum;                      // expected-error{{'_Complex _Accum' is invalid}}
_Complex long _Accum cmplx_s_long_accum;            // expected-error{{'_Complex _Accum' is invalid}}

/* Bad combinations */
float _Accum f_accum;     // expected-error{{cannot combine with previous 'float' declaration specifier}}
double _Accum d_accum;    // expected-error{{cannot combine with previous 'double' declaration specifier}}
_Bool _Accum b_accum;     // expected-error{{cannot combine with previous '_Bool' declaration specifier}}
char _Accum c_accum;      // expected-error{{cannot combine with previous 'char' declaration specifier}}
int _Accum i_accum;       // expected-error{{cannot combine with previous 'int' declaration specifier}}
