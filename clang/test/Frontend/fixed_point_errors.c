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


/* Literals that cannot fit into types */
signed short _Accum s_short_accum = 256.0hk;            // expected-error{{this value is too large for this fixed point type}}
unsigned short _Accum u_short_accum = 256.0uhk;         // expected-error{{this value is too large for this fixed point type}}
signed _Accum s_accum = 65536.0k;                       // expected-error{{this value is too large for this fixed point type}}
unsigned _Accum u_accum = 65536.0uk;                    // expected-error{{this value is too large for this fixed point type}}
signed long _Accum s_long_accum = 4294967296.0lk;       // expected-error{{this value is too large for this fixed point type}}
unsigned long _Accum u_long_accum = 4294967296.0ulk;    // expected-error{{this value is too large for this fixed point type}}

// Large values from decimal exponents
short _Accum          short_accum_exp   = 2.56e2hk;           // expected-error{{this value is too large for this fixed point type}}
_Accum                accum_exp         = 6.5536e4k;          // expected-error{{this value is too large for this fixed point type}}
long _Accum           long_accum_exp    = 4.294967296e9lk;    // expected-error{{this value is too large for this fixed point type}}
unsigned short _Accum u_short_accum_exp = 2.56e2uhk;          // expected-error{{this value is too large for this fixed point type}}
unsigned _Accum       u_accum_exp       = 6.5536e4uk;         // expected-error{{this value is too large for this fixed point type}}
unsigned long _Accum  u_long_accum_exp  = 4.294967296e9ulk;   // expected-error{{this value is too large for this fixed point type}}

// Large value from hexidecimal exponents
short _Accum          short_accum_hex_exp   = 0x1p8hk;        // expected-error{{this value is too large for this fixed point type}}
_Accum                accum_hex_exp         = 0x1p16k;        // expected-error{{this value is too large for this fixed point type}}
long _Accum           long_accum_hex_exp    = 0x1p32lk;       // expected-error{{this value is too large for this fixed point type}}
unsigned short _Accum u_short_accum_hex_exp = 0x1p8uhk;       // expected-error{{this value is too large for this fixed point type}}
unsigned _Accum       u_accum_hex_exp       = 0x1p16uk;       // expected-error{{this value is too large for this fixed point type}}
unsigned long _Accum  u_long_accum_hex_exp  = 0x1p32ulk;      // expected-error{{this value is too large for this fixed point type}}

// Very large exponent
_Accum x = 1e1000000000000000000000000000000000k;   // expected-error{{this value is too large for this fixed point type}}

/* Although _Fract's cannot equal 1, _Fract literals written as 1 are allowed
 * and the underlying value represents the max value for that _Fract type. */
short _Fract          short_fract_above_1    = 1.1hr;   // expected-error{{this value is too large for this fixed point type}}
_Fract                fract_above_1          = 1.1r;    // expected-error{{this value is too large for this fixed point type}}
long _Fract           long_fract_above_1     = 1.1lr;   // expected-error{{this value is too large for this fixed point type}}
unsigned short _Fract u_short_fract_above_1  = 1.1uhr;  // expected-error{{this value is too large for this fixed point type}}
unsigned _Fract       u_fract_above_1        = 1.1ur;   // expected-error{{this value is too large for this fixed point type}}
unsigned long _Fract  u_long_fract_above_1   = 1.1ulr;  // expected-error{{this value is too large for this fixed point type}}

short _Fract          short_fract_hex_exp   = 0x0.fp1hr;      // expected-error{{this value is too large for this fixed point type}}
_Fract                fract_hex_exp         = 0x0.fp1r;       // expected-error{{this value is too large for this fixed point type}}
long _Fract           long_fract_hex_exp    = 0x0.fp1lr;      // expected-error{{this value is too large for this fixed point type}}
unsigned short _Fract u_short_fract_hex_exp = 0x0.fp1uhr;     // expected-error{{this value is too large for this fixed point type}}
unsigned _Fract       u_fract_hex_exp       = 0x0.fp1ur;      // expected-error{{this value is too large for this fixed point type}}
unsigned long _Fract  u_long_fract_hex_exp  = 0x0.fp1ulr;     // expected-error{{this value is too large for this fixed point type}}

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

/* Bad suffixes  */
_Accum fk = 1.0fk; // expected-error{{invalid suffix 'fk' on fixed-point constant}}
_Accum kk = 1.0kk; // expected-error{{invalid suffix 'kk' on fixed-point constant}}
_Accum rk = 1.0rk; // expected-error{{invalid suffix 'rk' on fixed-point constant}}
_Accum rk = 1.0rr; // expected-error{{invalid suffix 'rr' on fixed-point constant}}
_Accum qk = 1.0qr; // expected-error{{invalid suffix 'qr' on fixed-point constant}}

/* Using wrong exponent notation */
_Accum dec_with_hex_exp1 = 0.1p10k;    // expected-error{{invalid suffix 'p10k' on fixed-point constant}}
_Accum dec_with_hex_exp2 = 0.1P10k;    // expected-error{{invalid suffix 'P10k' on fixed-point constant}}
_Accum hex_with_dex_exp1 = 0x0.1e10k;  // expected-error{{hexadecimal floating constant requires an exponent}}
_Accum hex_with_dex_exp2 = 0x0.1E10k;  // expected-error{{hexadecimal floating constant requires an exponent}}

void CheckSuffixOnIntegerLiterals() {
  _Accum short_acc_int;
  _Accum acc_int;
  _Accum long_acc_int;

  _Accum u_short_acc_int;
  _Accum u_acc_int;
  _Accum u_long_acc_int;

  _Fract short_fract_int;
  _Fract fract_int;
  _Fract long_fract_int;

  _Fract u_short_fract_int;
  _Fract u_fract_int;
  _Fract u_long_fract_int;

  // Decimal integer literals (non-zero)
  short_acc_int = 10hk; // expected-error{{invalid suffix 'hk' on integer constant}}
  acc_int = 10k;        // expected-error{{invalid suffix 'k' on integer constant}}
  long_acc_int = 10lk;  // expected-error{{invalid suffix 'lk' on integer constant}}

  u_short_acc_int = 10uhk; // expected-error{{invalid suffix 'uhk' on integer constant}}
  u_acc_int = 10uk;        // expected-error{{invalid suffix 'uk' on integer constant}}
  u_long_acc_int = 10ulk;  // expected-error{{invalid suffix 'ulk' on integer constant}}

  short_fract_int = 10hr; // expected-error{{invalid suffix 'hr' on integer constant}}
  fract_int = 10r;        // expected-error{{invalid suffix 'r' on integer constant}}
  long_fract_int = 10lr;  // expected-error{{invalid suffix 'lr' on integer constant}}

  u_short_fract_int = 10uhr; // expected-error{{invalid suffix 'uhr' on integer constant}}
  u_fract_int = 10ur;        // expected-error{{invalid suffix 'ur' on integer constant}}
  u_long_fract_int = 10ulr;  // expected-error{{invalid suffix 'ulr' on integer constant}}

  // Decimal integer literals (0)
  short_acc_int = 0hk; // expected-error{{invalid suffix 'hk' on integer constant}}
  acc_int = 0k;        // expected-error{{invalid suffix 'k' on integer constant}}
  long_acc_int = 0lk;  // expected-error{{invalid suffix 'lk' on integer constant}}

  // Decimal integer literals (large number)
  acc_int = 999999999999999999k;   // expected-error{{invalid suffix 'k' on integer constant}}
  fract_int = 999999999999999999r; // expected-error{{invalid suffix 'r' on integer constant}}

  // Octal integer literals
  short_acc_int = 010hk; // expected-error{{invalid suffix 'hk' on integer constant}}
  acc_int = 010k;        // expected-error{{invalid suffix 'k' on integer constant}}
  long_acc_int = 010lk;  // expected-error{{invalid suffix 'lk' on integer constant}}

  u_short_acc_int = 010uhk; // expected-error{{invalid suffix 'uhk' on integer constant}}
  u_acc_int = 010uk;        // expected-error{{invalid suffix 'uk' on integer constant}}
  u_long_acc_int = 010ulk;  // expected-error{{invalid suffix 'ulk' on integer constant}}

  short_fract_int = 010hr; // expected-error{{invalid suffix 'hr' on integer constant}}
  fract_int = 010r;        // expected-error{{invalid suffix 'r' on integer constant}}
  long_fract_int = 010lr;  // expected-error{{invalid suffix 'lr' on integer constant}}

  u_short_fract_int = 010uhr; // expected-error{{invalid suffix 'uhr' on integer constant}}
  u_fract_int = 010ur;        // expected-error{{invalid suffix 'ur' on integer constant}}
  u_long_fract_int = 010ulr;  // expected-error{{invalid suffix 'ulr' on integer constant}}

  // Hexadecimal integer literals
  short_acc_int = 0x10hk; // expected-error{{invalid suffix 'hk' on integer constant}}
  acc_int = 0x10k;        // expected-error{{invalid suffix 'k' on integer constant}}
  long_acc_int = 0x10lk;  // expected-error{{invalid suffix 'lk' on integer constant}}

  u_short_acc_int = 0x10uhk; // expected-error{{invalid suffix 'uhk' on integer constant}}
  u_acc_int = 0x10uk;        // expected-error{{invalid suffix 'uk' on integer constant}}
  u_long_acc_int = 0x10ulk;  // expected-error{{invalid suffix 'ulk' on integer constant}}

  short_fract_int = 0x10hr; // expected-error{{invalid suffix 'hr' on integer constant}}
  fract_int = 0x10r;        // expected-error{{invalid suffix 'r' on integer constant}}
  long_fract_int = 0x10lr;  // expected-error{{invalid suffix 'lr' on integer constant}}

  u_short_fract_int = 0x10uhr; // expected-error{{invalid suffix 'uhr' on integer constant}}
  u_fract_int = 0x10ur;        // expected-error{{invalid suffix 'ur' on integer constant}}
  u_long_fract_int = 0x10ulr;  // expected-error{{invalid suffix 'ulr' on integer constant}}

  // Using auto
  auto auto_fract = 0r;  // expected-error{{invalid suffix 'r' on integer constant}}
                         // expected-warning@-1{{type specifier missing, defaults to 'int'}}
  auto auto_accum = 0k;  // expected-error{{invalid suffix 'k' on integer constant}}
                         // expected-warning@-1{{type specifier missing, defaults to 'int'}}
}

// Ok conversions
int i_const = -2.5hk;
_Sat short _Accum sat_sa_const2 = 256.0k;
_Sat unsigned short _Accum sat_usa_const = -1.0hk;
short _Accum sa_const3 = 2;
short _Accum sa_const4 = -2;

// Overflow
short _Accum sa_const = 256.0k;   // expected-warning{{implicit conversion from 256.0 cannot fit within the range of values for 'short _Accum'}}
short _Fract sf_const = 1.0hk;    // expected-warning{{implicit conversion from 1.0 cannot fit within the range of values for 'short _Fract'}}
unsigned _Accum ua_const = -1.0k; // expected-warning{{implicit conversion from -1.0 cannot fit within the range of values for 'unsigned _Accum'}}
short _Accum sa_const2 = 128.0k + 128.0k; // expected-warning{{implicit conversion from 256.0 cannot fit within the range of values for 'short _Accum'}}
short s_const = 65536.0lk;                // expected-warning{{implicit conversion from 65536.0 cannot fit within the range of values for 'short'}}
unsigned u_const = -2.5hk;                // expected-warning{{implicit conversion from -2.5 cannot fit within the range of values for 'unsigned int'}}
char c_const = 256.0uk;                   // expected-warning{{implicit conversion from 256.0 cannot fit within the range of values for 'char'}}
short _Accum sa_const5 = 256;             // expected-warning{{implicit conversion from 256 cannot fit within the range of values for 'short _Accum'}}
unsigned short _Accum usa_const2 = -2;    // expected-warning{{implicit conversion from -2 cannot fit within the range of values for 'unsigned short _Accum'}}

short _Accum add_ovf1 = 255.0hk + 20.0hk;                     // expected-warning {{overflow in expression; result is -237.0 with type 'short _Accum'}}
short _Accum add_ovf2 = 10 + 0.5hr;                           // expected-warning {{overflow in expression; result is 0.5 with type 'short _Fract'}}
short _Accum sub_ovf1 = 16.0uhk - 32.0uhk;                    // expected-warning {{overflow in expression; result is 240.0 with type 'unsigned short _Accum'}}
short _Accum sub_ovf2 = -255.0hk - 20;                        // expected-warning {{overflow in expression; result is 237.0 with type 'short _Accum'}}
short _Accum mul_ovf1 = 200.0uhk * 10.0uhk;                   // expected-warning {{overflow in expression; result is 208.0 with type 'unsigned short _Accum'}}
short _Accum mul_ovf2 = (-0.5hr - 0.5hr) * (-0.5hr - 0.5hr);  // expected-warning {{overflow in expression; result is -1.0 with type 'short _Fract'}}
short _Accum div_ovf1 = 255.0hk / 0.5hk;                      // expected-warning {{overflow in expression; result is -2.0 with type 'short _Accum'}}

short _Accum shl_ovf1 = 255.0hk << 8;           // expected-warning {{overflow in expression; result is -256.0 with type 'short _Accum'}}
short _Fract shl_ovf2 = -0.25hr << 3;           // expected-warning {{overflow in expression; result is 0.0 with type 'short _Fract'}}
unsigned short _Accum shl_ovf3 = 100.5uhk << 3; // expected-warning {{overflow in expression; result is 36.0 with type 'unsigned short _Accum'}}
short _Fract shl_ovf4 = 0.25hr << 2;            // expected-warning {{overflow in expression; result is -1.0 with type 'short _Fract'}}

_Accum shl_bw1 = 0.000091552734375k << 32;                   // expected-warning {{shift count >= width of type}} \
                                                                 expected-warning {{overflow in expression; result is -65536.0 with type '_Accum'}}
unsigned _Fract shl_bw2 = 0.65ur << 16;                      // expected-warning {{shift count >= width of type}} \
                                                                 expected-warning {{overflow in expression; result is 0.0 with type 'unsigned _Fract'}}
_Sat short _Accum shl_bw3 = (_Sat short _Accum)80.0hk << 17; // expected-warning {{shift count >= width of type}}
short _Accum shr_bw1 = 1.0hk >> 17;                          // expected-warning {{shift count >= width of type}}

_Accum shl_neg1 = 25.5k << -5;  // expected-warning {{shift count is negative}} \
                                                              // expected-warning {{overflow in expression; result is 0.0 with type '_Accum'}}
_Accum shr_neg1 = 8.75k >> -9;  // expected-warning {{shift count is negative}}
_Fract shl_neg2 = 0.25r << -17; // expected-warning {{shift count is negative}} \
                                                              // expected-warning {{overflow in expression; result is 0.0 with type '_Fract'}}

// No warnings for saturation
short _Fract add_sat  = (_Sat short _Fract)0.5hr + 0.5hr;
short _Accum sub_sat  = (_Sat short _Accum)-200.0hk - 80.0hk;
short _Accum mul_sat  = (_Sat short _Accum)80.0hk * 10.0hk;
short _Fract div_sat  = (_Sat short _Fract)0.9hr / 0.1hr;
short _Accum shl_sat = (_Sat short _Accum)200.0hk << 5;

// Division by zero
short _Accum div_zero = 4.5k / 0.0lr;  // expected-error {{initializer element is not a compile-time constant}}
