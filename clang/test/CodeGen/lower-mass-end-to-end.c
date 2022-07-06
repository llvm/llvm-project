// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass -O3 -fapprox-func --target=powerpc64le-unknown-linux-gnu -S %s -o -| FileCheck %s -check-prefix=CHECK-MASS-AFN
// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass -Ofast --target=powerpc64le-unknown-linux-gnu -S %s -o -| FileCheck %s -check-prefix=CHECK-MASS-FAST
// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass -O3 -fapprox-func --target=powerpc-ibm-aix-xcoff -S %s -o -| FileCheck %s -check-prefix=CHECK-MASS-AFN
// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass -Ofast --target=powerpc-ibm-aix-xcoff -S %s -o -| FileCheck %s -check-prefix=CHECK-MASS-FAST

// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass=false -O3 -fapprox-func --target=powerpc64le-unknown-linux-gnu -S %s -o -| FileCheck %s -check-prefix=CHECK-NO-MASS-AFN
// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass=false -Ofast --target=powerpc64le-unknown-linux-gnu -S %s -o -| FileCheck %s -check-prefix=CHECK-NO-MASS-FAST
// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass=false -O3 -fapprox-func --target=powerpc-ibm-aix-xcoff -S %s -o -| FileCheck %s -check-prefix=CHECK-NO-MASS-AFN
// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass=false -Ofast --target=powerpc-ibm-aix-xcoff -S %s -o -| FileCheck %s -check-prefix=CHECK-NO-MASS-FAST

// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass -O3 -fno-approx-func --target=powerpc64le-unknown-linux-gnu -S %s -o -| FileCheck %s -check-prefix=CHECK-NO-MASS-AFN
// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass -fno-fast-math --target=powerpc64le-unknown-linux-gnu -S %s -o -| FileCheck %s -check-prefix=CHECK-NO-MASS-FAST
// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass -O3 -fno-approx-func --target=powerpc-ibm-aix-xcoff -S %s -o -| FileCheck %s -check-prefix=CHECK-NO-MASS-AFN
// RUN: %clang -mllvm -enable-ppc-gen-scalar-mass -fno-fast-math --target=powerpc-ibm-aix-xcoff -S %s -o -| FileCheck %s -check-prefix=CHECK-NO-MASS-FAST

extern double sin(double a);
extern double cos(double a);
extern double pow(double a, double b);
extern double log(double a);
extern double log10(double a);
extern double exp(double a);
extern float sinf(float a);
extern float cosf(float a);
extern float powf(float a, float b);
extern float logf(float a);
extern float log10f(float a);
extern float expf(float a);

double sin_f64(double a) {
// CHECK-LABEL: sin_f64
// CHECK-MASS-FAST: __xl_sin_finite
// CHECK-MASS-AFN: __xl_sin
// CHECK-NO-MASS-FAST-NOT: {{__xl_sin|__xl_sin_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_sin|__xl_sin_finite}}
// CHECK: blr
  return sin(a);
}

double cos_f64(double a) {
// CHECK-LABEL: cos_f64
// CHECK-MASS-FAST: __xl_cos_finite
// CHECK-MASS-AFN: __xl_cos
// CHECK-NO-MASS-FAST-NOT: {{__xl_cos|__xl_cos_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_cos|__xl_cos_finite}}
// CHECK: blr
  return cos(a);
}

double pow_f64(double a, double b) {
// CHECK-LABEL: pow_f64
// CHECK-MASS-FAST: __xl_pow_finite
// CHECK-MASS-AFN: __xl_pow
// CHECK-NO-MASS-FAST-NOT: {{__xl_pow|__xl_pow_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_pow|__xl_pow_finite}}
// CHECK: blr
  return pow(a, b);
}

double log_f64(double a) {
// CHECK-LABEL: log_f64
// CHECK-MASS-FAST: __xl_log_finite
// CHECK-MASS-AFN: __xl_log
// CHECK-NO-MASS-FAST-NOT: {{__xl_log|__xl_log_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_log|__xl_log_finite}}
// CHECK: blr
  return log(a);
}

double log10_f64(double a) {
// CHECK-LABEL: log10_f64
// CHECK-MASS-FAST: __xl_log10_finite
// CHECK-MASS-AFN: __xl_log10
// CHECK-NO-MASS-FAST-NOT: {{__xl_log10|__xl_log10_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_log10|__xl_log10_finite}}
// CHECK: blr
  return log10(a);
}

double exp_f64(double a) {
// CHECK-LABEL: exp_f64
// CHECK-MASS-FAST: __xl_exp_finite
// CHECK-MASS-AFN: __xl_exp
// CHECK-NO-MASS-FAST-NOT: {{__xl_exp|__xl_exp_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_exp|__xl_exp_finite}}
// CHECK: blr
  return exp(a);
}

float sin_f32(float a) {
// CHECK-LABEL: sin_f32
// CHECK-MASS-FAST: __xl_sinf_finite
// CHECK-MASS-AFN: __xl_sinf
// CHECK-NO-MASS-FAST-NOT: {{__xl_sinf|__xl_sinf_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_sinf|__xl_sinf_finite}}
// CHECK: blr
  return sinf(a);
}

float cos_f32(float a) {
// CHECK-LABEL: cos_f32
// CHECK-MASS-FAST: __xl_cosf_finite
// CHECK-MASS-AFN: __xl_cosf
// CHECK-NO-MASS-FAST-NOT: {{__xl_cosf|__xl_cosf_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_cosf|__xl_cosf_finite}}
// CHECK: blr
  return cosf(a);
}

float pow_f32(float a, float b) {
// CHECK-LABEL: pow_f32
// CHECK-MASS-FAST: __xl_powf_finite
// CHECK-MASS-AFN: __xl_powf
// CHECK-NO-MASS-FAST-NOT: {{__xl_pow|__xl_powf_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_pow|__xl_powf_finite}}
// CHECK: blr
  return powf(a, b);
}

float log_f32(float a) {
// CHECK-LABEL: log_f32
// CHECK-MASS-FAST: __xl_logf_finite
// CHECK-MASS-AFN: __xl_logf
// CHECK-NO-MASS-FAST-NOT: {{__xl_logf|__xl_logf_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_logf|__xl_logf_finite}}
// CHECK: blr
  return logf(a);
}

float log10_f32(float a) {
// CHECK-LABEL: log10_f32
// CHECK-MASS-FAST: __xl_log10f_finite
// CHECK-MASS-AFN: __xl_log10f
// CHECK-NO-MASS-FAST-NOT: {{__xl_log10f|__xl_log10f_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_log10f|__xl_log10f_finite}}
// CHECK: blr
  return log10f(a);
}

float exp_f32(float a) {
// CHECK-LABEL: exp_f32
// CHECK-MASS-FAST: __xl_expf_finite
// CHECK-MASS-AFN: __xl_expf
// CHECK-NO-MASS-FAST-NOT: {{__xl_expf|__xl_expf_finite}}
// CHECK-NO-MASS-AFN-NOT: {{__xl_expf|__xl_expf_finite}}
// CHECK: blr
  return expf(a);
}
