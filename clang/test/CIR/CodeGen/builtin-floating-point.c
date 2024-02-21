// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffast-math -fclangir -emit-cir %s -o - | FileCheck %s

// ceil

float my_ceilf(float f) {
  return __builtin_ceilf(f);
  // CHECK: cir.func @my_ceilf
  // CHECK: {{.+}} = cir.ceil {{.+}} : !cir.float
}

double my_ceil(double f) {
  return __builtin_ceil(f);
  // CHECK: cir.func @my_ceil
  // CHECK: {{.+}} = cir.ceil {{.+}} : !cir.double
}

// long double my_ceill(long double f) {
//   return __builtin_ceill(f);
//   // DISABLED-CHECK: cir.func @my_ceill
//   // DISABLED-CHECK: {{.+}} = cir.ceil {{.+}} : f80
// }

float ceilf(float);
double ceil(double);
// long double ceill(long double);

float call_ceilf(float f) {
  return ceilf(f);
  // CHECK: cir.func @call_ceilf
  // CHECK: {{.+}} = cir.ceil {{.+}} : !cir.float
}

double call_ceil(double f) {
  return ceil(f);
  // CHECK: cir.func @call_ceil
  // CHECK: {{.+}} = cir.ceil {{.+}} : !cir.double
}

// long double call_ceill(long double f) {
//   return ceill(f);
//   // DISABLED-CHECK: cir.func @call_ceill
//   // DISABLED-CHECK: {{.+}} = cir.ceil {{.+}} : f80
// }

// cos

float my_cosf(float f) {
  return __builtin_cosf(f);
  // CHECK: cir.func @my_cosf
  // CHECK: {{.+}} = cir.cos {{.+}} : !cir.float
}

double my_cos(double f) {
  return __builtin_cos(f);
  // CHECK: cir.func @my_cos
  // CHECK: {{.+}} = cir.cos {{.+}} : !cir.double
}

// long double my_cosl(long double f) {
//   return __builtin_cosl(f);
//   // DISABLED-CHECK: cir.func @my_cosl
//   // DISABLED-CHECK: {{.+}} = cir.cos {{.+}} : f80
// }

float cosf(float);
double cos(double);
// long double cosl(long double);

float call_cosf(float f) {
  return cosf(f);
  // CHECK: cir.func @call_cosf
  // CHECK: {{.+}} = cir.cos {{.+}} : !cir.float
}

double call_cos(double f) {
  return cos(f);
  // CHECK: cir.func @call_cos
  // CHECK: {{.+}} = cir.cos {{.+}} : !cir.double
}

// long double call_cosl(long double f) {
//   return cosl(f);
//   // DISABLED-CHECK: cir.func @call_cosl
//   // DISABLED-CHECK: {{.+}} = cir.cos {{.+}} : f80
// }

// exp

float my_expf(float f) {
  return __builtin_expf(f);
  // CHECK: cir.func @my_expf
  // CHECK: {{.+}} = cir.exp {{.+}} : !cir.float
}

double my_exp(double f) {
  return __builtin_exp(f);
  // CHECK: cir.func @my_exp
  // CHECK: {{.+}} = cir.exp {{.+}} : !cir.double
}

// long double my_expl(long double f) {
//   return __builtin_expl(f);
//   // DISABLED-CHECK: cir.func @my_expl
//   // DISABLED-CHECK: {{.+}} = cir.exp {{.+}} : f80
// }

float expf(float);
double exp(double);
// long double expl(long double);

float call_expf(float f) {
  return expf(f);
  // CHECK: cir.func @call_expf
  // CHECK: {{.+}} = cir.exp {{.+}} : !cir.float
}

double call_exp(double f) {
  return exp(f);
  // CHECK: cir.func @call_exp
  // CHECK: {{.+}} = cir.exp {{.+}} : !cir.double
}

// long double call_expl(long double f) {
//   return expl(f);
//   // DISABLED-CHECK: cir.func @call_expl
//   // DISABLED-CHECK: {{.+}} = cir.exp {{.+}} : f80
// }

// exp2

float my_exp2f(float f) {
  return __builtin_exp2f(f);
  // CHECK: cir.func @my_exp2f
  // CHECK: {{.+}} = cir.exp2 {{.+}} : !cir.float
}

double my_exp2(double f) {
  return __builtin_exp2(f);
  // CHECK: cir.func @my_exp2
  // CHECK: {{.+}} = cir.exp2 {{.+}} : !cir.double
}

// long double my_exp2l(long double f) {
//   return __builtin_exp2l(f);
//   // DISABLED-CHECK: cir.func @my_exp2l
//   // DISABLED-CHECK: {{.+}} = cir.exp2 {{.+}} : f80
// }

float exp2f(float);
double exp2(double);
// long double exp2l(long double);

float call_exp2f(float f) {
  return exp2f(f);
  // CHECK: cir.func @call_exp2f
  // CHECK: {{.+}} = cir.exp2 {{.+}} : !cir.float
}

double call_exp2(double f) {
  return exp2(f);
  // CHECK: cir.func @call_exp2
  // CHECK: {{.+}} = cir.exp2 {{.+}} : !cir.double
}

// long double call_exp2l(long double f) {
//   return exp2l(f);
//   // DISABLED-CHECK: cir.func @call_exp2l
//   // DISABLED-CHECK: {{.+}} = cir.exp2 {{.+}} : f80
// }

// floor

float my_floorf(float f) {
  return __builtin_floorf(f);
  // CHECK: cir.func @my_floorf
  // CHECK: {{.+}} = cir.floor {{.+}} : !cir.float
}

double my_floor(double f) {
  return __builtin_floor(f);
  // CHECK: cir.func @my_floor
  // CHECK: {{.+}} = cir.floor {{.+}} : !cir.double
}

// long double my_floorl(long double f) {
//   return __builtin_floorl(f);
//   // DISABLED-CHECK: cir.func @my_floorl
//   // DISABLED-CHECK: {{.+}} = cir.floor {{.+}} : f80
// }

float floorf(float);
double floor(double);
// long double floorl(long double);

float call_floorf(float f) {
  return floorf(f);
  // CHECK: cir.func @call_floorf
  // CHECK: {{.+}} = cir.floor {{.+}} : !cir.float
}

double call_floor(double f) {
  return floor(f);
  // CHECK: cir.func @call_floor
  // CHECK: {{.+}} = cir.floor {{.+}} : !cir.double
}

// long double call_floorl(long double f) {
//   return floorl(f);
//   // DISABLED-CHECK: cir.func @call_floorl
//   // DISABLED-CHECK: {{.+}} = cir.floor {{.+}} : f80
// }

// log

float my_logf(float f) {
  return __builtin_logf(f);
  // CHECK: cir.func @my_logf
  // CHECK: {{.+}} = cir.log {{.+}} : !cir.float
}

double my_log(double f) {
  return __builtin_log(f);
  // CHECK: cir.func @my_log
  // CHECK: {{.+}} = cir.log {{.+}} : !cir.double
}

// long double my_logl(long double f) {
//   return __builtin_logl(f);
//   // DISABLED-CHECK: cir.func @my_logl
//   // DISABLED-CHECK: {{.+}} = cir.log {{.+}} : f80
// }

float logf(float);
double log(double);
// long double logl(long double);

float call_logf(float f) {
  return logf(f);
  // CHECK: cir.func @call_logf
  // CHECK: {{.+}} = cir.log {{.+}} : !cir.float
}

double call_log(double f) {
  return log(f);
  // CHECK: cir.func @call_log
  // CHECK: {{.+}} = cir.log {{.+}} : !cir.double
}

// long double call_logl(long double f) {
//   return logl(f);
//   // DISABLED-CHECK: cir.func @call_logl
//   // DISABLED-CHECK: {{.+}} = cir.log {{.+}} : f80
// }

// log10

float my_log10f(float f) {
  return __builtin_log10f(f);
  // CHECK: cir.func @my_log10f
  // CHECK: {{.+}} = cir.log10 {{.+}} : !cir.float
}

double my_log10(double f) {
  return __builtin_log10(f);
  // CHECK: cir.func @my_log10
  // CHECK: {{.+}} = cir.log10 {{.+}} : !cir.double
}

// long double my_log10l(long double f) {
//   return __builtin_log10l(f);
//   // DISABLED-CHECK: cir.func @my_log10l
//   // DISABLED-CHECK: {{.+}} = cir.log10 {{.+}} : f80
// }

float log10f(float);
double log10(double);
// long double log10l(long double);

float call_log10f(float f) {
  return log10f(f);
  // CHECK: cir.func @call_log10f
  // CHECK: {{.+}} = cir.log10 {{.+}} : !cir.float
}

double call_log10(double f) {
  return log10(f);
  // CHECK: cir.func @call_log10
  // CHECK: {{.+}} = cir.log10 {{.+}} : !cir.double
}

// long double call_log10l(long double f) {
//   return log10l(f);
//   // DISABLED-CHECK: cir.func @call_log10l
//   // DISABLED-CHECK: {{.+}} = cir.log10 {{.+}} : f80
// }

// log2

float my_log2f(float f) {
  return __builtin_log2f(f);
  // CHECK: cir.func @my_log2f
  // CHECK: {{.+}} = cir.log2 {{.+}} : !cir.float
}

double my_log2(double f) {
  return __builtin_log2(f);
  // CHECK: cir.func @my_log2
  // CHECK: {{.+}} = cir.log2 {{.+}} : !cir.double
}

// long double my_log2l(long double f) {
//   return __builtin_log2l(f);
//   // DISABLED-CHECK: cir.func @my_log2l
//   // DISABLED-CHECK: {{.+}} = cir.log2 {{.+}} : f80
// }

float log2f(float);
double log2(double);
// long double log2l(long double);

float call_log2f(float f) {
  return log2f(f);
  // CHECK: cir.func @call_log2f
  // CHECK: {{.+}} = cir.log2 {{.+}} : !cir.float
}

double call_log2(double f) {
  return log2(f);
  // CHECK: cir.func @call_log2
  // CHECK: {{.+}} = cir.log2 {{.+}} : !cir.double
}

// long double call_log2l(long double f) {
//   return log2l(f);
//   // DISABLED-CHECK: cir.func @call_log2l
//   // DISABLED-CHECK: {{.+}} = cir.log2 {{.+}} : f80
// }

// nearbyint

float my_nearbyintf(float f) {
  return __builtin_nearbyintf(f);
  // CHECK: cir.func @my_nearbyintf
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : !cir.float
}

double my_nearbyint(double f) {
  return __builtin_nearbyint(f);
  // CHECK: cir.func @my_nearbyint
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : !cir.double
}

// long double my_nearbyintl(long double f) {
//   return __builtin_nearbyintl(f);
//   // DISABLED-CHECK: cir.func @my_nearbyintl
//   // DISABLED-CHECK: {{.+}} = cir.nearbyint {{.+}} : f80
// }

float nearbyintf(float);
double nearbyint(double);
// long double nearbyintl(long double);

float call_nearbyintf(float f) {
  return nearbyintf(f);
  // CHECK: cir.func @call_nearbyintf
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : !cir.float
}

double call_nearbyint(double f) {
  return nearbyint(f);
  // CHECK: cir.func @call_nearbyint
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : !cir.double
}

// long double call_nearbyintl(long double f) {
//   return nearbyintl(f);
//   // DISABLED-CHECK: cir.func @call_nearbyintl
//   // DISABLED-CHECK: {{.+}} = cir.nearbyint {{.+}} : f80
// }

// rint

float my_rintf(float f) {
  return __builtin_rintf(f);
  // CHECK: cir.func @my_rintf
  // CHECK: {{.+}} = cir.rint {{.+}} : !cir.float
}

double my_rint(double f) {
  return __builtin_rint(f);
  // CHECK: cir.func @my_rint
  // CHECK: {{.+}} = cir.rint {{.+}} : !cir.double
}

// long double my_rintl(long double f) {
//   return __builtin_rintl(f);
//   // DISABLED-CHECK: cir.func @my_rintl
//   // DISABLED-CHECK: {{.+}} = cir.rint {{.+}} : f80
// }

float rintf(float);
double rint(double);
// long double rintl(long double);

float call_rintf(float f) {
  return rintf(f);
  // CHECK: cir.func @call_rintf
  // CHECK: {{.+}} = cir.rint {{.+}} : !cir.float
}

double call_rint(double f) {
  return rint(f);
  // CHECK: cir.func @call_rint
  // CHECK: {{.+}} = cir.rint {{.+}} : !cir.double
}

// long double call_rintl(long double f) {
//   return rintl(f);
//   // DISABLED-CHECK: cir.func @call_rintl
//   // DISABLED-CHECK: {{.+}} = cir.rint {{.+}} : f80
// }

// round

float my_roundf(float f) {
  return __builtin_roundf(f);
  // CHECK: cir.func @my_roundf
  // CHECK: {{.+}} = cir.round {{.+}} : !cir.float
}

double my_round(double f) {
  return __builtin_round(f);
  // CHECK: cir.func @my_round
  // CHECK: {{.+}} = cir.round {{.+}} : !cir.double
}

// long double my_roundl(long double f) {
//   return __builtin_roundl(f);
//   // DISABLED-CHECK: cir.func @my_roundl
//   // DISABLED-CHECK: {{.+}} = cir.round {{.+}} : f80
// }

float roundf(float);
double round(double);
// long double roundl(long double);

float call_roundf(float f) {
  return roundf(f);
  // CHECK: cir.func @call_roundf
  // CHECK: {{.+}} = cir.round {{.+}} : !cir.float
}

double call_round(double f) {
  return round(f);
  // CHECK: cir.func @call_round
  // CHECK: {{.+}} = cir.round {{.+}} : !cir.double
}

// long double call_roundl(long double f) {
//   return roundl(f);
//   // DISABLED-CHECK: cir.func @call_roundl
//   // DISABLED-CHECK: {{.+}} = cir.round {{.+}} : f80
// }

// sin

float my_sinf(float f) {
  return __builtin_sinf(f);
  // CHECK: cir.func @my_sinf
  // CHECK: {{.+}} = cir.sin {{.+}} : !cir.float
}

double my_sin(double f) {
  return __builtin_sin(f);
  // CHECK: cir.func @my_sin
  // CHECK: {{.+}} = cir.sin {{.+}} : !cir.double
}

// long double my_sinl(long double f) {
//   return __builtin_sinl(f);
//   // DISABLED-CHECK: cir.func @my_sinl
//   // DISABLED-CHECK: {{.+}} = cir.sin {{.+}} : f80
// }

float sinf(float);
double sin(double);
// long double sinl(long double);

float call_sinf(float f) {
  return sinf(f);
  // CHECK: cir.func @call_sinf
  // CHECK: {{.+}} = cir.sin {{.+}} : !cir.float
}

double call_sin(double f) {
  return sin(f);
  // CHECK: cir.func @call_sin
  // CHECK: {{.+}} = cir.sin {{.+}} : !cir.double
}

// long double call_sinl(long double f) {
//   return sinl(f);
//   // DISABLED-CHECK: cir.func @call_sinl
//   // DISABLED-CHECK: {{.+}} = cir.sin {{.+}} : f80
// }

// sqrt

float my_sqrtf(float f) {
  return __builtin_sqrtf(f);
  // CHECK: cir.func @my_sqrtf
  // CHECK: {{.+}} = cir.sqrt {{.+}} : !cir.float
}

double my_sqrt(double f) {
  return __builtin_sqrt(f);
  // CHECK: cir.func @my_sqrt
  // CHECK: {{.+}} = cir.sqrt {{.+}} : !cir.double
}

// long double my_sqrtl(long double f) {
//   return __builtin_sqrtl(f);
//   // DISABLED-CHECK: cir.func @my_sqrtl
//   // DISABLED-CHECK: {{.+}} = cir.sqrt {{.+}} : f80
// }

float sqrtf(float);
double sqrt(double);
// long double sqrtl(long double);

float call_sqrtf(float f) {
  return sqrtf(f);
  // CHECK: cir.func @call_sqrtf
  // CHECK: {{.+}} = cir.sqrt {{.+}} : !cir.float
}

double call_sqrt(double f) {
  return sqrt(f);
  // CHECK: cir.func @call_sqrt
  // CHECK: {{.+}} = cir.sqrt {{.+}} : !cir.double
}

// long double call_sqrtl(long double f) {
//   return sqrtl(f);
//   // DISABLED-CHECK: cir.func @call_sqrtl
//   // DISABLED-CHECK: {{.+}} = cir.sqrt {{.+}} : f80
// }

// trunc

float my_truncf(float f) {
  return __builtin_truncf(f);
  // CHECK: cir.func @my_truncf
  // CHECK: {{.+}} = cir.trunc {{.+}} : !cir.float
}

double my_trunc(double f) {
  return __builtin_trunc(f);
  // CHECK: cir.func @my_trunc
  // CHECK: {{.+}} = cir.trunc {{.+}} : !cir.double
}

// long double my_truncl(long double f) {
//   return __builtin_truncl(f);
//   // DISABLED-CHECK: cir.func @my_truncl
//   // DISABLED-CHECK: {{.+}} = cir.trunc {{.+}} : f80
// }

float truncf(float);
double trunc(double);
// long double truncl(long double);

float call_truncf(float f) {
  return truncf(f);
  // CHECK: cir.func @call_truncf
  // CHECK: {{.+}} = cir.trunc {{.+}} : !cir.float
}

double call_trunc(double f) {
  return trunc(f);
  // CHECK: cir.func @call_trunc
  // CHECK: {{.+}} = cir.trunc {{.+}} : !cir.double
}

// long double call_truncl(long double f) {
//   return truncl(f);
//   // DISABLED-CHECK: cir.func @call_truncl
//   // DISABLED-CHECK: {{.+}} = cir.trunc {{.+}} : f80
// }
