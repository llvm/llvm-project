// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffast-math -fclangir -emit-cir %s -o - | FileCheck %s

// ceil

float my_ceilf(float f) {
  return __builtin_ceilf(f);
  // CHECK: cir.func @my_ceilf
  // CHECK: {{.+}} = cir.ceil {{.+}} : f32
}

double my_ceil(double f) {
  return __builtin_ceil(f);
  // CHECK: cir.func @my_ceil
  // CHECK: {{.+}} = cir.ceil {{.+}} : f64
}

long double my_ceill(long double f) {
  return __builtin_ceill(f);
  // CHECK: cir.func @my_ceill
  // CHECK: {{.+}} = cir.ceil {{.+}} : f80
}

float ceilf(float);
double ceil(double);
long double ceill(long double);

float call_ceilf(float f) {
  return ceilf(f);
  // CHECK: cir.func @call_ceilf
  // CHECK: {{.+}} = cir.ceil {{.+}} : f32
}

double call_ceil(double f) {
  return ceil(f);
  // CHECK: cir.func @call_ceil
  // CHECK: {{.+}} = cir.ceil {{.+}} : f64
}

long double call_ceill(long double f) {
  return ceill(f);
  // CHECK: cir.func @call_ceill
  // CHECK: {{.+}} = cir.ceil {{.+}} : f80
}

// cos

float my_cosf(float f) {
  return __builtin_cosf(f);
  // CHECK: cir.func @my_cosf
  // CHECK: {{.+}} = cir.cos {{.+}} : f32
}

double my_cos(double f) {
  return __builtin_cos(f);
  // CHECK: cir.func @my_cos
  // CHECK: {{.+}} = cir.cos {{.+}} : f64
}

long double my_cosl(long double f) {
  return __builtin_cosl(f);
  // CHECK: cir.func @my_cosl
  // CHECK: {{.+}} = cir.cos {{.+}} : f80
}

float cosf(float);
double cos(double);
long double cosl(long double);

float call_cosf(float f) {
  return cosf(f);
  // CHECK: cir.func @call_cosf
  // CHECK: {{.+}} = cir.cos {{.+}} : f32
}

double call_cos(double f) {
  return cos(f);
  // CHECK: cir.func @call_cos
  // CHECK: {{.+}} = cir.cos {{.+}} : f64
}

long double call_cosl(long double f) {
  return cosl(f);
  // CHECK: cir.func @call_cosl
  // CHECK: {{.+}} = cir.cos {{.+}} : f80
}

// exp

float my_expf(float f) {
  return __builtin_expf(f);
  // CHECK: cir.func @my_expf
  // CHECK: {{.+}} = cir.exp {{.+}} : f32
}

double my_exp(double f) {
  return __builtin_exp(f);
  // CHECK: cir.func @my_exp
  // CHECK: {{.+}} = cir.exp {{.+}} : f64
}

long double my_expl(long double f) {
  return __builtin_expl(f);
  // CHECK: cir.func @my_expl
  // CHECK: {{.+}} = cir.exp {{.+}} : f80
}

float expf(float);
double exp(double);
long double expl(long double);

float call_expf(float f) {
  return expf(f);
  // CHECK: cir.func @call_expf
  // CHECK: {{.+}} = cir.exp {{.+}} : f32
}

double call_exp(double f) {
  return exp(f);
  // CHECK: cir.func @call_exp
  // CHECK: {{.+}} = cir.exp {{.+}} : f64
}

long double call_expl(long double f) {
  return expl(f);
  // CHECK: cir.func @call_expl
  // CHECK: {{.+}} = cir.exp {{.+}} : f80
}

// exp2

float my_exp2f(float f) {
  return __builtin_exp2f(f);
  // CHECK: cir.func @my_exp2f
  // CHECK: {{.+}} = cir.exp2 {{.+}} : f32
}

double my_exp2(double f) {
  return __builtin_exp2(f);
  // CHECK: cir.func @my_exp2
  // CHECK: {{.+}} = cir.exp2 {{.+}} : f64
}

long double my_exp2l(long double f) {
  return __builtin_exp2l(f);
  // CHECK: cir.func @my_exp2l
  // CHECK: {{.+}} = cir.exp2 {{.+}} : f80
}

float exp2f(float);
double exp2(double);
long double exp2l(long double);

float call_exp2f(float f) {
  return exp2f(f);
  // CHECK: cir.func @call_exp2f
  // CHECK: {{.+}} = cir.exp2 {{.+}} : f32
}

double call_exp2(double f) {
  return exp2(f);
  // CHECK: cir.func @call_exp2
  // CHECK: {{.+}} = cir.exp2 {{.+}} : f64
}

long double call_exp2l(long double f) {
  return exp2l(f);
  // CHECK: cir.func @call_exp2l
  // CHECK: {{.+}} = cir.exp2 {{.+}} : f80
}

// floor

float my_floorf(float f) {
  return __builtin_floorf(f);
  // CHECK: cir.func @my_floorf
  // CHECK: {{.+}} = cir.floor {{.+}} : f32
}

double my_floor(double f) {
  return __builtin_floor(f);
  // CHECK: cir.func @my_floor
  // CHECK: {{.+}} = cir.floor {{.+}} : f64
}

long double my_floorl(long double f) {
  return __builtin_floorl(f);
  // CHECK: cir.func @my_floorl
  // CHECK: {{.+}} = cir.floor {{.+}} : f80
}

float floorf(float);
double floor(double);
long double floorl(long double);

float call_floorf(float f) {
  return floorf(f);
  // CHECK: cir.func @call_floorf
  // CHECK: {{.+}} = cir.floor {{.+}} : f32
}

double call_floor(double f) {
  return floor(f);
  // CHECK: cir.func @call_floor
  // CHECK: {{.+}} = cir.floor {{.+}} : f64
}

long double call_floorl(long double f) {
  return floorl(f);
  // CHECK: cir.func @call_floorl
  // CHECK: {{.+}} = cir.floor {{.+}} : f80
}

// log

float my_logf(float f) {
  return __builtin_logf(f);
  // CHECK: cir.func @my_logf
  // CHECK: {{.+}} = cir.log {{.+}} : f32
}

double my_log(double f) {
  return __builtin_log(f);
  // CHECK: cir.func @my_log
  // CHECK: {{.+}} = cir.log {{.+}} : f64
}

long double my_logl(long double f) {
  return __builtin_logl(f);
  // CHECK: cir.func @my_logl
  // CHECK: {{.+}} = cir.log {{.+}} : f80
}

float logf(float);
double log(double);
long double logl(long double);

float call_logf(float f) {
  return logf(f);
  // CHECK: cir.func @call_logf
  // CHECK: {{.+}} = cir.log {{.+}} : f32
}

double call_log(double f) {
  return log(f);
  // CHECK: cir.func @call_log
  // CHECK: {{.+}} = cir.log {{.+}} : f64
}

long double call_logl(long double f) {
  return logl(f);
  // CHECK: cir.func @call_logl
  // CHECK: {{.+}} = cir.log {{.+}} : f80
}

// log10

float my_log10f(float f) {
  return __builtin_log10f(f);
  // CHECK: cir.func @my_log10f
  // CHECK: {{.+}} = cir.log10 {{.+}} : f32
}

double my_log10(double f) {
  return __builtin_log10(f);
  // CHECK: cir.func @my_log10
  // CHECK: {{.+}} = cir.log10 {{.+}} : f64
}

long double my_log10l(long double f) {
  return __builtin_log10l(f);
  // CHECK: cir.func @my_log10l
  // CHECK: {{.+}} = cir.log10 {{.+}} : f80
}

float log10f(float);
double log10(double);
long double log10l(long double);

float call_log10f(float f) {
  return log10f(f);
  // CHECK: cir.func @call_log10f
  // CHECK: {{.+}} = cir.log10 {{.+}} : f32
}

double call_log10(double f) {
  return log10(f);
  // CHECK: cir.func @call_log10
  // CHECK: {{.+}} = cir.log10 {{.+}} : f64
}

long double call_log10l(long double f) {
  return log10l(f);
  // CHECK: cir.func @call_log10l
  // CHECK: {{.+}} = cir.log10 {{.+}} : f80
}

// log2

float my_log2f(float f) {
  return __builtin_log2f(f);
  // CHECK: cir.func @my_log2f
  // CHECK: {{.+}} = cir.log2 {{.+}} : f32
}

double my_log2(double f) {
  return __builtin_log2(f);
  // CHECK: cir.func @my_log2
  // CHECK: {{.+}} = cir.log2 {{.+}} : f64
}

long double my_log2l(long double f) {
  return __builtin_log2l(f);
  // CHECK: cir.func @my_log2l
  // CHECK: {{.+}} = cir.log2 {{.+}} : f80
}

float log2f(float);
double log2(double);
long double log2l(long double);

float call_log2f(float f) {
  return log2f(f);
  // CHECK: cir.func @call_log2f
  // CHECK: {{.+}} = cir.log2 {{.+}} : f32
}

double call_log2(double f) {
  return log2(f);
  // CHECK: cir.func @call_log2
  // CHECK: {{.+}} = cir.log2 {{.+}} : f64
}

long double call_log2l(long double f) {
  return log2l(f);
  // CHECK: cir.func @call_log2l
  // CHECK: {{.+}} = cir.log2 {{.+}} : f80
}

// nearbyint

float my_nearbyintf(float f) {
  return __builtin_nearbyintf(f);
  // CHECK: cir.func @my_nearbyintf
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : f32
}

double my_nearbyint(double f) {
  return __builtin_nearbyint(f);
  // CHECK: cir.func @my_nearbyint
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : f64
}

long double my_nearbyintl(long double f) {
  return __builtin_nearbyintl(f);
  // CHECK: cir.func @my_nearbyintl
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : f80
}

float nearbyintf(float);
double nearbyint(double);
long double nearbyintl(long double);

float call_nearbyintf(float f) {
  return nearbyintf(f);
  // CHECK: cir.func @call_nearbyintf
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : f32
}

double call_nearbyint(double f) {
  return nearbyint(f);
  // CHECK: cir.func @call_nearbyint
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : f64
}

long double call_nearbyintl(long double f) {
  return nearbyintl(f);
  // CHECK: cir.func @call_nearbyintl
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : f80
}

// rint

float my_rintf(float f) {
  return __builtin_rintf(f);
  // CHECK: cir.func @my_rintf
  // CHECK: {{.+}} = cir.rint {{.+}} : f32
}

double my_rint(double f) {
  return __builtin_rint(f);
  // CHECK: cir.func @my_rint
  // CHECK: {{.+}} = cir.rint {{.+}} : f64
}

long double my_rintl(long double f) {
  return __builtin_rintl(f);
  // CHECK: cir.func @my_rintl
  // CHECK: {{.+}} = cir.rint {{.+}} : f80
}

float rintf(float);
double rint(double);
long double rintl(long double);

float call_rintf(float f) {
  return rintf(f);
  // CHECK: cir.func @call_rintf
  // CHECK: {{.+}} = cir.rint {{.+}} : f32
}

double call_rint(double f) {
  return rint(f);
  // CHECK: cir.func @call_rint
  // CHECK: {{.+}} = cir.rint {{.+}} : f64
}

long double call_rintl(long double f) {
  return rintl(f);
  // CHECK: cir.func @call_rintl
  // CHECK: {{.+}} = cir.rint {{.+}} : f80
}

// round

float my_roundf(float f) {
  return __builtin_roundf(f);
  // CHECK: cir.func @my_roundf
  // CHECK: {{.+}} = cir.round {{.+}} : f32
}

double my_round(double f) {
  return __builtin_round(f);
  // CHECK: cir.func @my_round
  // CHECK: {{.+}} = cir.round {{.+}} : f64
}

long double my_roundl(long double f) {
  return __builtin_roundl(f);
  // CHECK: cir.func @my_roundl
  // CHECK: {{.+}} = cir.round {{.+}} : f80
}

float roundf(float);
double round(double);
long double roundl(long double);

float call_roundf(float f) {
  return roundf(f);
  // CHECK: cir.func @call_roundf
  // CHECK: {{.+}} = cir.round {{.+}} : f32
}

double call_round(double f) {
  return round(f);
  // CHECK: cir.func @call_round
  // CHECK: {{.+}} = cir.round {{.+}} : f64
}

long double call_roundl(long double f) {
  return roundl(f);
  // CHECK: cir.func @call_roundl
  // CHECK: {{.+}} = cir.round {{.+}} : f80
}

// sin

float my_sinf(float f) {
  return __builtin_sinf(f);
  // CHECK: cir.func @my_sinf
  // CHECK: {{.+}} = cir.sin {{.+}} : f32
}

double my_sin(double f) {
  return __builtin_sin(f);
  // CHECK: cir.func @my_sin
  // CHECK: {{.+}} = cir.sin {{.+}} : f64
}

long double my_sinl(long double f) {
  return __builtin_sinl(f);
  // CHECK: cir.func @my_sinl
  // CHECK: {{.+}} = cir.sin {{.+}} : f80
}

float sinf(float);
double sin(double);
long double sinl(long double);

float call_sinf(float f) {
  return sinf(f);
  // CHECK: cir.func @call_sinf
  // CHECK: {{.+}} = cir.sin {{.+}} : f32
}

double call_sin(double f) {
  return sin(f);
  // CHECK: cir.func @call_sin
  // CHECK: {{.+}} = cir.sin {{.+}} : f64
}

long double call_sinl(long double f) {
  return sinl(f);
  // CHECK: cir.func @call_sinl
  // CHECK: {{.+}} = cir.sin {{.+}} : f80
}

// sqrt

float my_sqrtf(float f) {
  return __builtin_sqrtf(f);
  // CHECK: cir.func @my_sqrtf
  // CHECK: {{.+}} = cir.sqrt {{.+}} : f32
}

double my_sqrt(double f) {
  return __builtin_sqrt(f);
  // CHECK: cir.func @my_sqrt
  // CHECK: {{.+}} = cir.sqrt {{.+}} : f64
}

long double my_sqrtl(long double f) {
  return __builtin_sqrtl(f);
  // CHECK: cir.func @my_sqrtl
  // CHECK: {{.+}} = cir.sqrt {{.+}} : f80
}

float sqrtf(float);
double sqrt(double);
long double sqrtl(long double);

float call_sqrtf(float f) {
  return sqrtf(f);
  // CHECK: cir.func @call_sqrtf
  // CHECK: {{.+}} = cir.sqrt {{.+}} : f32
}

double call_sqrt(double f) {
  return sqrt(f);
  // CHECK: cir.func @call_sqrt
  // CHECK: {{.+}} = cir.sqrt {{.+}} : f64
}

long double call_sqrtl(long double f) {
  return sqrtl(f);
  // CHECK: cir.func @call_sqrtl
  // CHECK: {{.+}} = cir.sqrt {{.+}} : f80
}

// trunc

float my_truncf(float f) {
  return __builtin_truncf(f);
  // CHECK: cir.func @my_truncf
  // CHECK: {{.+}} = cir.trunc {{.+}} : f32
}

double my_trunc(double f) {
  return __builtin_trunc(f);
  // CHECK: cir.func @my_trunc
  // CHECK: {{.+}} = cir.trunc {{.+}} : f64
}

long double my_truncl(long double f) {
  return __builtin_truncl(f);
  // CHECK: cir.func @my_truncl
  // CHECK: {{.+}} = cir.trunc {{.+}} : f80
}

float truncf(float);
double trunc(double);
long double truncl(long double);

float call_truncf(float f) {
  return truncf(f);
  // CHECK: cir.func @call_truncf
  // CHECK: {{.+}} = cir.trunc {{.+}} : f32
}

double call_trunc(double f) {
  return trunc(f);
  // CHECK: cir.func @call_trunc
  // CHECK: {{.+}} = cir.trunc {{.+}} : f64
}

long double call_truncl(long double f) {
  return truncl(f);
  // CHECK: cir.func @call_truncl
  // CHECK: {{.+}} = cir.trunc {{.+}} : f80
}
