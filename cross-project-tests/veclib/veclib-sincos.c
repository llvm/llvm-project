// REQUIRES: aarch64-registered-target
// RUN: %clang -S -target aarch64-unknown-linux-gnu -O2 -fno-math-errno \
// RUN:  -fveclib=ArmPL -o - %s | FileCheck -check-prefix=ARMPL %s
// RUN: %clang -S -target aarch64-unknown-linux-gnu -O2 -fno-math-errno \
// RUN:  -fveclib=SLEEF -o - %s | FileCheck -check-prefix=SLEEF %s

typedef __SIZE_TYPE__ size_t;

void sincos(double, double *, double *);

// ARMPL: armpl_vcexpiq_f64
// ARMPL: armpl_vcexpiq_f64

// SLEEF: _ZGVnN2vl8l8_sincos
// SLEEF: _ZGVnN2vl8l8_sincos
void vectorize_sincos(double *restrict x, double *restrict s,
                      double *restrict c, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    sincos(x[i], &s[i], &c[i]);
  }
}
