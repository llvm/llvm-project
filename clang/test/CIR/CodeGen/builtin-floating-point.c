// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffast-math -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t1.cir 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-apple-darwin-macho -ffast-math -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t1.cir 2>&1 | FileCheck %s --check-prefix=AARCH64

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

long double my_ceill(long double f) {
  return __builtin_ceill(f);
  // CHECK: cir.func @my_ceill
  // CHECK: {{.+}} = cir.ceil {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.ceil {{.+}} : !cir.long_double<!cir.double>
}

float ceilf(float);
double ceil(double);
long double ceill(long double);

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

long double call_ceill(long double f) {
  return ceill(f);
  // CHECK: cir.func @call_ceill
  // CHECK: {{.+}} = cir.ceil {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.ceil {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_cosl(long double f) {
  return __builtin_cosl(f);
  // CHECK: cir.func @my_cosl
  // CHECK: {{.+}} = cir.cos {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.cos {{.+}} : !cir.long_double<!cir.double>
}

float cosf(float);
double cos(double);
long double cosl(long double);

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

long double call_cosl(long double f) {
  return cosl(f);
  // CHECK: cir.func @call_cosl
  // CHECK: {{.+}} = cir.cos {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.cos {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_expl(long double f) {
  return __builtin_expl(f);
  // CHECK: cir.func @my_expl
  // CHECK: {{.+}} = cir.exp {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.exp {{.+}} : !cir.long_double<!cir.double>
}

float expf(float);
double exp(double);
long double expl(long double);

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

long double call_expl(long double f) {
  return expl(f);
  // CHECK: cir.func @call_expl
  // CHECK: {{.+}} = cir.exp {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.exp {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_exp2l(long double f) {
  return __builtin_exp2l(f);
  // CHECK: cir.func @my_exp2l
  // CHECK: {{.+}} = cir.exp2 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.exp2 {{.+}} : !cir.long_double<!cir.double>
}

float exp2f(float);
double exp2(double);
long double exp2l(long double);

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

long double call_exp2l(long double f) {
  return exp2l(f);
  // CHECK: cir.func @call_exp2l
  // CHECK: {{.+}} = cir.exp2 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.exp2 {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_floorl(long double f) {
  return __builtin_floorl(f);
  // CHECK: cir.func @my_floorl
  // CHECK: {{.+}} = cir.floor {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.floor {{.+}} : !cir.long_double<!cir.double>
}

float floorf(float);
double floor(double);
long double floorl(long double);

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

long double call_floorl(long double f) {
  return floorl(f);
  // CHECK: cir.func @call_floorl
  // CHECK: {{.+}} = cir.floor {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.floor {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_logl(long double f) {
  return __builtin_logl(f);
  // CHECK: cir.func @my_logl
  // CHECK: {{.+}} = cir.log {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log {{.+}} : !cir.long_double<!cir.double>
}

float logf(float);
double log(double);
long double logl(long double);

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

long double call_logl(long double f) {
  return logl(f);
  // CHECK: cir.func @call_logl
  // CHECK: {{.+}} = cir.log {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_log10l(long double f) {
  return __builtin_log10l(f);
  // CHECK: cir.func @my_log10l
  // CHECK: {{.+}} = cir.log10 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log10 {{.+}} : !cir.long_double<!cir.double>
}

float log10f(float);
double log10(double);
long double log10l(long double);

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

long double call_log10l(long double f) {
  return log10l(f);
  // CHECK: cir.func @call_log10l
  // CHECK: {{.+}} = cir.log10 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log10 {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_log2l(long double f) {
  return __builtin_log2l(f);
  // CHECK: cir.func @my_log2l
  // CHECK: {{.+}} = cir.log2 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log2 {{.+}} : !cir.long_double<!cir.double>
}

float log2f(float);
double log2(double);
long double log2l(long double);

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

long double call_log2l(long double f) {
  return log2l(f);
  // CHECK: cir.func @call_log2l
  // CHECK: {{.+}} = cir.log2 {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.log2 {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_nearbyintl(long double f) {
  return __builtin_nearbyintl(f);
  // CHECK: cir.func @my_nearbyintl
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.nearbyint {{.+}} : !cir.long_double<!cir.double>
}

float nearbyintf(float);
double nearbyint(double);
long double nearbyintl(long double);

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

long double call_nearbyintl(long double f) {
  return nearbyintl(f);
  // CHECK: cir.func @call_nearbyintl
  // CHECK: {{.+}} = cir.nearbyint {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.nearbyint {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_rintl(long double f) {
  return __builtin_rintl(f);
  // CHECK: cir.func @my_rintl
  // CHECK: {{.+}} = cir.rint {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.rint {{.+}} : !cir.long_double<!cir.double>
}

float rintf(float);
double rint(double);
long double rintl(long double);

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

long double call_rintl(long double f) {
  return rintl(f);
  // CHECK: cir.func @call_rintl
  // CHECK: {{.+}} = cir.rint {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.rint {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_roundl(long double f) {
  return __builtin_roundl(f);
  // CHECK: cir.func @my_roundl
  // CHECK: {{.+}} = cir.round {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.round {{.+}} : !cir.long_double<!cir.double>
}

float roundf(float);
double round(double);
long double roundl(long double);

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

long double call_roundl(long double f) {
  return roundl(f);
  // CHECK: cir.func @call_roundl
  // CHECK: {{.+}} = cir.round {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.round {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_sinl(long double f) {
  return __builtin_sinl(f);
  // CHECK: cir.func @my_sinl
  // CHECK: {{.+}} = cir.sin {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.sin {{.+}} : !cir.long_double<!cir.double>
}

float sinf(float);
double sin(double);
long double sinl(long double);

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

long double call_sinl(long double f) {
  return sinl(f);
  // CHECK: cir.func @call_sinl
  // CHECK: {{.+}} = cir.sin {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.sin {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_sqrtl(long double f) {
  return __builtin_sqrtl(f);
  // CHECK: cir.func @my_sqrtl
  // CHECK: {{.+}} = cir.sqrt {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.sqrt {{.+}} : !cir.long_double<!cir.double>
}

float sqrtf(float);
double sqrt(double);
long double sqrtl(long double);

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

long double call_sqrtl(long double f) {
  return sqrtl(f);
  // CHECK: cir.func @call_sqrtl
  // CHECK: {{.+}} = cir.sqrt {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.sqrt {{.+}} : !cir.long_double<!cir.double>
}

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

long double my_truncl(long double f) {
  return __builtin_truncl(f);
  // CHECK: cir.func @my_truncl
  // CHECK: {{.+}} = cir.trunc {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.trunc {{.+}} : !cir.long_double<!cir.double>
}

float truncf(float);
double trunc(double);
long double truncl(long double);

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

long double call_truncl(long double f) {
  return truncl(f);
  // CHECK: cir.func @call_truncl
  // CHECK: {{.+}} = cir.trunc {{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: {{.+}} = cir.trunc {{.+}} : !cir.long_double<!cir.double>
}

// copysign

float my_copysignf(float x, float y) {
  return __builtin_copysignf(x, y);
  // CHECK: cir.func @my_copysignf
  // CHECK:   %{{.+}} = cir.copysign %{{.+}}, %{{.+}} : !cir.float
}

double my_copysign(double x, double y) {
  return __builtin_copysign(x, y);
  // CHECK: cir.func @my_copysign
  // CHECK:   %{{.+}} = cir.copysign %{{.+}}, %{{.+}} : !cir.double
}

long double my_copysignl(long double x, long double y) {
  return __builtin_copysignl(x, y);
  // CHECK: cir.func @my_copysignl
  // CHECK:   %{{.+}} = cir.copysign %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: %{{.+}} = cir.copysign %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>
}

float copysignf(float, float);
double copysign(double, double);
long double copysignl(long double, long double);

float call_copysignf(float x, float y) {
  return copysignf(x, y);
  // CHECK: cir.func @call_copysignf
  // CHECK:   %{{.+}} = cir.copysign %{{.+}}, %{{.+}} : !cir.float
}

double call_copysign(double x, double y) {
  return copysign(x, y);
  // CHECK: cir.func @call_copysign
  // CHECK:   %{{.+}} = cir.copysign %{{.+}}, %{{.+}} : !cir.double
}

long double call_copysignl(long double x, long double y) {
  return copysignl(x, y);
  // CHECK: cir.func @call_copysignl
  // CHECK:   %{{.+}} = cir.copysign %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: %{{.+}} = cir.copysign %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>
}

// fmax

float my_fmaxf(float x, float y) {
  return __builtin_fmaxf(x, y);
  // CHECK: cir.func @my_fmaxf
  // CHECK:   %{{.+}} = cir.fmax %{{.+}}, %{{.+}} : !cir.float
}

double my_fmax(double x, double y) {
  return __builtin_fmax(x, y);
  // CHECK: cir.func @my_fmax
  // CHECK:   %{{.+}} = cir.fmax %{{.+}}, %{{.+}} : !cir.double
}

long double my_fmaxl(long double x, long double y) {
  return __builtin_fmaxl(x, y);
  // CHECK: cir.func @my_fmaxl
  // CHECK:   %{{.+}} = cir.fmax %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: %{{.+}} = cir.fmax %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>
}

float fmaxf(float, float);
double fmax(double, double);
long double fmaxl(long double, long double);

float call_fmaxf(float x, float y) {
  return fmaxf(x, y);
  // CHECK: cir.func @call_fmaxf
  // CHECK:   %{{.+}} = cir.fmax %{{.+}}, %{{.+}} : !cir.float
}

double call_fmax(double x, double y) {
  return fmax(x, y);
  // CHECK: cir.func @call_fmax
  // CHECK:   %{{.+}} = cir.fmax %{{.+}}, %{{.+}} : !cir.double
}

long double call_fmaxl(long double x, long double y) {
  return fmaxl(x, y);
  // CHECK: cir.func @call_fmaxl
  // CHECK:   %{{.+}} = cir.fmax %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: %{{.+}} = cir.fmax %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>
}

// fmin

float my_fminf(float x, float y) {
  return __builtin_fminf(x, y);
  // CHECK: cir.func @my_fminf
  // CHECK:   %{{.+}} = cir.fmin %{{.+}}, %{{.+}} : !cir.float
}

double my_fmin(double x, double y) {
  return __builtin_fmin(x, y);
  // CHECK: cir.func @my_fmin
  // CHECK:   %{{.+}} = cir.fmin %{{.+}}, %{{.+}} : !cir.double
}

long double my_fminl(long double x, long double y) {
  return __builtin_fminl(x, y);
  // CHECK: cir.func @my_fminl
  // CHECK:   %{{.+}} = cir.fmin %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: %{{.+}} = cir.fmin %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>
}

float fminf(float, float);
double fmin(double, double);
long double fminl(long double, long double);

float call_fminf(float x, float y) {
  return fminf(x, y);
  // CHECK: cir.func @call_fminf
  // CHECK:   %{{.+}} = cir.fmin %{{.+}}, %{{.+}} : !cir.float
}

double call_fmin(double x, double y) {
  return fmin(x, y);
  // CHECK: cir.func @call_fmin
  // CHECK:   %{{.+}} = cir.fmin %{{.+}}, %{{.+}} : !cir.double
}

long double call_fminl(long double x, long double y) {
  return fminl(x, y);
  // CHECK: cir.func @call_fminl
  // CHECK:   %{{.+}} = cir.fmin %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: %{{.+}} = cir.fmin %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>
}

// fmod

float my_fmodf(float x, float y) {
  return __builtin_fmodf(x, y);
  // CHECK: cir.func @my_fmodf
  // CHECK:   %{{.+}} = cir.fmod %{{.+}}, %{{.+}} : !cir.float
}

double my_fmod(double x, double y) {
  return __builtin_fmod(x, y);
  // CHECK: cir.func @my_fmod
  // CHECK:   %{{.+}} = cir.fmod %{{.+}}, %{{.+}} : !cir.double
}

long double my_fmodl(long double x, long double y) {
  return __builtin_fmodl(x, y);
  // CHECK: cir.func @my_fmodl
  // CHECK:   %{{.+}} = cir.fmod %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: %{{.+}} = cir.fmod %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>
}

float fmodf(float, float);
double fmod(double, double);
long double fmodl(long double, long double);

float call_fmodf(float x, float y) {
  return fmodf(x, y);
  // CHECK: cir.func @call_fmodf
  // CHECK:   %{{.+}} = cir.fmod %{{.+}}, %{{.+}} : !cir.float
}

double call_fmod(double x, double y) {
  return fmod(x, y);
  // CHECK: cir.func @call_fmod
  // CHECK:   %{{.+}} = cir.fmod %{{.+}}, %{{.+}} : !cir.double
}

long double call_fmodl(long double x, long double y) {
  return fmodl(x, y);
  // CHECK: cir.func @call_fmodl
  // CHECK:   %{{.+}} = cir.fmod %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: %{{.+}} = cir.fmod %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>
}

// pow

float my_powf(float x, float y) {
  return __builtin_powf(x, y);
  // CHECK: cir.func @my_powf
  // CHECK:   %{{.+}} = cir.pow %{{.+}}, %{{.+}} : !cir.float
}

double my_pow(double x, double y) {
  return __builtin_pow(x, y);
  // CHECK: cir.func @my_pow
  // CHECK:   %{{.+}} = cir.pow %{{.+}}, %{{.+}} : !cir.double
}

long double my_powl(long double x, long double y) {
  return __builtin_powl(x, y);
  // CHECK: cir.func @my_powl
  // CHECK:   %{{.+}} = cir.pow %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: %{{.+}} = cir.pow %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>
}

float powf(float, float);
double pow(double, double);
long double powl(long double, long double);

float call_powf(float x, float y) {
  return powf(x, y);
  // CHECK: cir.func @call_powf
  // CHECK:   %{{.+}} = cir.pow %{{.+}}, %{{.+}} : !cir.float
}

double call_pow(double x, double y) {
  return pow(x, y);
  // CHECK: cir.func @call_pow
  // CHECK:   %{{.+}} = cir.pow %{{.+}}, %{{.+}} : !cir.double
}

long double call_powl(long double x, long double y) {
  return powl(x, y);
  // CHECK: cir.func @call_powl
  // CHECK:   %{{.+}} = cir.pow %{{.+}}, %{{.+}} : !cir.long_double<!cir.f80>
  // AARCH64: %{{.+}} = cir.pow %{{.+}}, %{{.+}} : !cir.long_double<!cir.double>
}
