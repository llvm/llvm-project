// RUN: %clang_cc1 -verify=no-diag %s
// RUN: %clang_cc1 -complex-range=promoted -triple x86_64-unknown-linux \
// RUN: -verify=no-diag %s
// RUN: %clang_cc1 -complex-range=promoted -triple x86_64-unknown-windows \
// RUN: -verify=div-precision %s
// no-diag-no-diagnostics

// This tests evaluation of _Complex arithmetic at compile time.

#define APPROX_EQ(a, b) (                             \
  __builtin_fabs(__real (a) - __real (b)) < 0.0001 && \
  __builtin_fabs(__imag (a) - __imag (b)) < 0.0001    \
)

#define EVAL(a, b) _Static_assert(a == b, "")
#define EVALF(a, b) _Static_assert(APPROX_EQ(a, b), "")

// _Complex float + _Complex float
void a() {
  EVALF((2.f + 3i) + (4.f + 5i), 6.f + 8i);
  EVALF((2.f + 3i) - (4.f + 5i), -2.f - 2i);
  EVALF((2.f + 3i) * (4.f + 5i), -7.f + 22i);
  EVALF((2.f + 3i) / (4.f + 5i), 0.5609f + 0.0487i);

  EVALF((2. + 3i) + (4. + 5i), 6. + 8i);
  EVALF((2. + 3i) - (4. + 5i), -2. - 2i);
  EVALF((2. + 3i) * (4. + 5i), -7. + 22i);
  // div-precision-warning@+1 {{excess precision is requested but the target does not support excess precision which may result in observable differences in complex division behavior, additional uses where the requested higher precision cannot be honored were found but not diagnosed}}
  EVALF((2. + 3i) / (4. + 5i), .5609 + .0487i);
}

// _Complex int + _Complex int
void b() {
  EVAL((2 + 3i) + (4 + 5i), 6 + 8i);
  EVAL((2 + 3i) - (4 + 5i), -2 - 2i);
  EVAL((2 + 3i) * (4 + 5i), -7 + 22i);
  EVAL((8 + 30i) / (4 + 5i), 4 + 1i);
}

// _Complex float + float
void c() {
  EVALF((2.f + 4i) + 3.f, 5.f + 4i);
  EVALF((2.f + 4i) - 3.f, -1.f + 4i);
  EVALF((2.f + 4i) * 3.f, 6.f + 12i);
  EVALF((2.f + 4i) / 2.f, 1.f + 2i);

  EVALF(3.f + (2.f + 4i), 5.f + 4i);
  EVALF(3.f - (2.f + 4i), 1.f - 4i);
  EVALF(3.f * (2.f + 4i), 6.f + 12i);
  EVALF(3.f / (2.f + 4i), .3f - 0.6i);

  EVALF((2. + 4i) + 3., 5. + 4i);
  EVALF((2. + 4i) - 3., -1. + 4i);
  EVALF((2. + 4i) * 3., 6. + 12i);
  EVALF((2. + 4i) / 2., 1. + 2i);

  EVALF(3. + (2. + 4i), 5. + 4i);
  EVALF(3. - (2. + 4i), 1. - 4i);
  EVALF(3. * (2. + 4i), 6. + 12i);
  EVALF(3. / (2. + 4i), .3 - 0.6i);
}

// _Complex int + int
void d() {
  EVAL((2 + 4i) + 3, 5 + 4i);
  EVAL((2 + 4i) - 3, -1 + 4i);
  EVAL((2 + 4i) * 3, 6 + 12i);
  EVAL((2 + 4i) / 2, 1 + 2i);

  EVAL(3 + (2 + 4i), 5 + 4i);
  EVAL(3 - (2 + 4i), 1 - 4i);
  EVAL(3 * (2 + 4i), 6 + 12i);
  EVAL(20 / (2 + 4i), 2 - 4i);
}

// _Complex float + int
void e() {
  EVALF((2.f + 4i) + 3, 5.f + 4i);
  EVALF((2.f + 4i) - 3, -1.f + 4i);
  EVALF((2.f + 4i) * 3, 6.f + 12i);
  EVALF((2.f + 4i) / 2, 1.f + 2i);

  EVALF(3 + (2.f + 4i), 5.f + 4i);
  EVALF(3 - (2.f + 4i), 1.f - 4i);
  EVALF(3 * (2.f + 4i), 6.f + 12i);
  EVALF(3 / (2.f + 4i), .3f - 0.6i);

  EVALF((2. + 4i) + 3, 5. + 4i);
  EVALF((2. + 4i) - 3, -1. + 4i);
  EVALF((2. + 4i) * 3, 6. + 12i);
  EVALF((2. + 4i) / 2, 1. + 2i);

  EVALF(3 + (2. + 4i), 5. + 4i);
  EVALF(3 - (2. + 4i), 1. - 4i);
  EVALF(3 * (2. + 4i), 6. + 12i);
  EVALF(3 / (2. + 4i), .3 - 0.6i);
}

// _Complex int + float
void f() {
  EVALF((2 + 4i) + 3.f, 5.f + 4i);
  EVALF((2 + 4i) - 3.f, -1.f + 4i);
  EVALF((2 + 4i) * 3.f, 6.f + 12i);
  EVALF((2 + 4i) / 2.f, 1.f + 2i);

  EVALF(3.f + (2 + 4i), 5.f + 4i);
  EVALF(3.f - (2 + 4i), 1.f - 4i);
  EVALF(3.f * (2 + 4i), 6.f + 12i);
  EVALF(3.f / (2 + 4i), .3f - 0.6i);

  EVALF((2 + 4i) + 3., 5. + 4i);
  EVALF((2 + 4i) - 3., -1. + 4i);
  EVALF((2 + 4i) * 3., 6. + 12i);
  EVALF((2 + 4i) / 2., 1. + 2i);

  EVALF(3. + (2 + 4i), 5. + 4i);
  EVALF(3. - (2 + 4i), 1. - 4i);
  EVALF(3. * (2 + 4i), 6. + 12i);
  EVALF(3. / (2 + 4i), .3 - 0.6i);
}
