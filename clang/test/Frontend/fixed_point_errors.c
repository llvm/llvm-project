// RUN: %clang_cc1 -verify -ffixed-point %s

/* We do not yet support long long. No recommended bit widths are given for this
 * size. */

long long _Accum longlong_accum;              // expected-error{{'long long _Accum' is invalid}}
unsigned long long _Accum u_longlong_accum;   // expected-error{{'long long _Accum' is invalid}}
long long _Fract longlong_fract;              // expected-error{{'long long _Fract' is invalid}}
unsigned long long _Fract u_longlong_fract;   // expected-error{{'long long _Fract' is invalid}}

_Sat long long _Accum sat_longlong_accum;             // expected-error{{'long long _Accum' is invalid}}
_Sat unsigned long long _Accum sat_u_longlong_accum;  // expected-error{{'long long _Accum' is invalid}}
_Sat long long _Fract sat_longlong_fract;             // expected-error{{'long long _Fract' is invalid}}
_Sat unsigned long long _Fract sat_u_longlong_fract;  // expected-error{{'long long _Fract' is invalid}}


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

_Complex signed short _Fract cmplx_s_short_fract;   // expected-error{{'_Complex _Fract' is invalid}}
_Complex signed _Fract cmplx_s_fract;               // expected-error{{'_Complex _Fract' is invalid}}
_Complex signed long _Fract cmplx_s_long_fract;     // expected-error{{'_Complex _Fract' is invalid}}
_Complex unsigned short _Fract cmplx_u_short_fract; // expected-error{{'_Complex _Fract' is invalid}}
_Complex unsigned _Fract cmplx_u_fract;             // expected-error{{'_Complex _Fract' is invalid}}
_Complex unsigned long _Fract cmplx_u_long_fract;   // expected-error{{'_Complex _Fract' is invalid}}
_Complex short _Fract cmplx_s_short_fract;          // expected-error{{'_Complex _Fract' is invalid}}
_Complex _Fract cmplx_s_fract;                      // expected-error{{'_Complex _Fract' is invalid}}
_Complex long _Fract cmplx_s_long_fract;            // expected-error{{'_Complex _Fract' is invalid}}

_Complex _Sat signed short _Accum cmplx_sat_s_short_accum;   // expected-error{{'_Complex _Accum' is invalid}}
_Complex _Sat signed _Accum cmplx_sat_s_accum;               // expected-error{{'_Complex _Accum' is invalid}}
_Complex _Sat signed long _Accum cmplx_sat_s_long_accum;     // expected-error{{'_Complex _Accum' is invalid}}
_Complex _Sat unsigned short _Accum cmplx_sat_u_short_accum; // expected-error{{'_Complex _Accum' is invalid}}
_Complex _Sat unsigned _Accum cmplx_sat_u_accum;             // expected-error{{'_Complex _Accum' is invalid}}
_Complex _Sat unsigned long _Accum cmplx_sat_u_long_accum;   // expected-error{{'_Complex _Accum' is invalid}}
_Complex _Sat short _Accum cmplx_sat_s_short_accum;          // expected-error{{'_Complex _Accum' is invalid}}
_Complex _Sat _Accum cmplx_sat_s_accum;                      // expected-error{{'_Complex _Accum' is invalid}}
_Complex _Sat long _Accum cmplx_sat_s_long_accum;            // expected-error{{'_Complex _Accum' is invalid}}

_Complex signed short _Fract cmplx_sat_s_short_fract;   // expected-error{{'_Complex _Fract' is invalid}}
_Complex signed _Fract cmplx_sat_s_fract;               // expected-error{{'_Complex _Fract' is invalid}}
_Complex signed long _Fract cmplx_sat_s_long_fract;     // expected-error{{'_Complex _Fract' is invalid}}
_Complex unsigned short _Fract cmplx_sat_u_short_fract; // expected-error{{'_Complex _Fract' is invalid}}
_Complex unsigned _Fract cmplx_sat_u_fract;             // expected-error{{'_Complex _Fract' is invalid}}
_Complex unsigned long _Fract cmplx_sat_u_long_fract;   // expected-error{{'_Complex _Fract' is invalid}}
_Complex short _Fract cmplx_sat_s_short_fract;          // expected-error{{'_Complex _Fract' is invalid}}
_Complex _Fract cmplx_sat_s_fract;                      // expected-error{{'_Complex _Fract' is invalid}}
_Complex long _Fract cmplx_sat_s_long_fract;            // expected-error{{'_Complex _Fract' is invalid}}

/* Bad combinations */
float _Accum f_accum;     // expected-error{{cannot combine with previous 'float' declaration specifier}}
double _Accum d_accum;    // expected-error{{cannot combine with previous 'double' declaration specifier}}
_Bool _Accum b_accum;     // expected-error{{cannot combine with previous '_Bool' declaration specifier}}
char _Accum c_accum;      // expected-error{{cannot combine with previous 'char' declaration specifier}}
int _Accum i_accum;       // expected-error{{cannot combine with previous 'int' declaration specifier}}

float _Fract f_fract;     // expected-error{{cannot combine with previous 'float' declaration specifier}}
double _Fract d_fract;    // expected-error{{cannot combine with previous 'double' declaration specifier}}
_Bool _Fract b_fract;     // expected-error{{cannot combine with previous '_Bool' declaration specifier}}
char _Fract c_fract;      // expected-error{{cannot combine with previous 'char' declaration specifier}}
int _Fract i_fract;       // expected-error{{cannot combine with previous 'int' declaration specifier}}

/* Bad saturated combinations */
_Sat float f;             // expected-error{{'_Sat' specifier is only valid on '_Fract' or '_Accum', not 'float'}}
_Sat double d;            // expected-error{{'_Sat' specifier is only valid on '_Fract' or '_Accum', not 'double'}}
_Sat _Bool b;             // expected-error{{'_Sat' specifier is only valid on '_Fract' or '_Accum', not '_Bool'}}
_Sat char c;              // expected-error{{'_Sat' specifier is only valid on '_Fract' or '_Accum', not 'char'}}
_Sat int i;               // expected-error{{'_Sat' specifier is only valid on '_Fract' or '_Accum', not 'int'}}
_Sat _Sat _Fract fract;   // expected-warning{{duplicate '_Sat' declaration specifier}}

/* Do not allow typedef to be used with typedef'd types */
typedef short _Fract shortfract_t;
typedef short _Accum shortaccum_t;
typedef _Fract fract_t;
typedef _Accum accum_t;
typedef long _Fract longfract_t;
typedef long _Accum longaccum_t;
_Sat shortfract_t td_sat_short_fract;       // expected-error{{'_Sat' specifier is only valid on '_Fract' or '_Accum', not 'type-name'}}
_Sat shortaccum_t td_sat_short_accum;       // expected-error{{'_Sat' specifier is only valid on '_Fract' or '_Accum', not 'type-name'}}
_Sat fract_t td_sat_fract;                  // expected-error{{'_Sat' specifier is only valid on '_Fract' or '_Accum', not 'type-name'}}
_Sat accum_t td_sat_accum;                  // expected-error{{'_Sat' specifier is only valid on '_Fract' or '_Accum', not 'type-name'}}
_Sat longfract_t td_sat_long_fract;         // expected-error{{'_Sat' specifier is only valid on '_Fract' or '_Accum', not 'type-name'}}
_Sat longaccum_t td_sat_long_accum;         // expected-error{{'_Sat' specifier is only valid on '_Fract' or '_Accum', not 'type-name'}}
