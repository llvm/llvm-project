// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -std=c++23 \
// RUN: -DWIN -emit-llvm -o - %s | FileCheck %s --check-prefixes=WIN

// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -std=c++23 \
// RUN: -emit-llvm -o - %s | FileCheck %s --check-prefixes=LNX

#ifdef WIN
#define INFINITY ((float)(1e+300 * 1e+300))
#define NAN      (-(float)(INFINITY * 0.0F))
#else
#define NAN (__builtin_nanf(""))
#define INFINITY (__builtin_inff())
#endif

int func()
{
  int i;

  // fmin
  constexpr double f1 = __builtin_fmin(15.24, 1.3);
  constexpr double f2 = __builtin_fmin(-0.0, +0.0);
  constexpr double f3 = __builtin_fmin(+0.0, -0.0);
  constexpr float f4 = __builtin_fminf(NAN, NAN);
  constexpr float f5 = __builtin_fminf(NAN, -1);
  constexpr float f6 = __builtin_fminf(-INFINITY, 0);
  constexpr float f7 = __builtin_fminf(INFINITY, 0);
  constexpr long double f8 = __builtin_fminl(123.456L, 789.012L);

  // frexp
  constexpr double f9 = __builtin_frexp(123.45, &i);
  constexpr double f10 = __builtin_frexp(0.0, &i);
  constexpr double f11 = __builtin_frexp(-0.0, &i);
  constexpr double f12 = __builtin_frexpf(NAN, &i);
  constexpr double f13 = __builtin_frexpf(-NAN, &i);
  constexpr double f14 = __builtin_frexpf(INFINITY, &i);
  constexpr double f15 = __builtin_frexpf(INFINITY, &i);
  constexpr long double f16 = __builtin_frexpl(259.328L, &i);

  return 0;
}

// CHECK: store double 1.300000e+00, ptr {{.*}}
// CHECK: store double -0.000000e+00, ptr {{.*}}
// CHECK: store double -0.000000e+00, ptr {{.*}}
// WIN: store float 0xFFF8000000000000, ptr {{.*}}
// LNX: store float 0x7FF8000000000000, ptr {{.*}}
// CHECK: store float -1.000000e+00, ptr {{.*}}
// CHECK: store float 0xFFF0000000000000, ptr {{.*}}
// CHECK: store double 1.234560e+02, ptr {{.*}}

// CHECK: store double 0x3FEEDCCCCCCCCCCD, ptr {{.*}}
// CHECK: store double 0.000000e+00, ptr {{.*}}
// CHECK: store double -0.000000e+00, ptr {{.*}}
// CHECK: store double 0xFFF8000000000000, ptr {{.*}}
// WIN: store double 0x7FF8000000000000, ptr {{.*}}
// LNX: store double 0xFFF8000000000000, ptr {{.*}}
// CHECK: store double 0x7FF0000000000000, ptr {{.*}}
// CHECK: store double 0x7FF0000000000000, ptr {{.*}}
// CHECK: store double 5.065000e-01, ptr {{.*}}
