// RUN: %clang_cc1 -emit-llvm -triple i386-linux -Wno-unknown-pragmas %s -o - | FileCheck %s

double sin(double);
double sin_rte(double);
double sin_rtz(double);
double sin_rtp(double);
double sin_rtn(double);
double sin_rta(double);

#define CONCAT(a, b) CONCAT_(a, b)
#define CONCAT_(a, b) a##b
#define ADD_ROUNDING_MODE_SUFFIX(func) CONCAT(func, __ROUNDING_MODE__)

#define sin(x) ADD_ROUNDING_MODE_SUFFIX(sin)(x)

double call_dyn(double x) {
  return sin(x);
}
// CHECK-LABEL: define {{.*}} double @call_dyn(
// CHECK:       call double @llvm.sin.f64(

#pragma STDC FENV_ROUND FE_TOWARDZERO
double call_tz(double x) {
  return sin(x);
}
// CHECK-LABEL: define {{.*}} double @call_tz(
// CHECK:       call double @sin_rtz(

#pragma STDC FENV_ROUND FE_TONEAREST
double call_te(double x) {
  return sin(x);
}
// CHECK-LABEL: define {{.*}} double @call_te(
// CHECK:       call double @sin_rte(

#pragma STDC FENV_ROUND FE_DOWNWARD
double call_tn(double x) {
  return sin(x);
}
// CHECK-LABEL: define {{.*}} double @call_tn(
// CHECK:       call double @sin_rtn(

#pragma STDC FENV_ROUND FE_UPWARD
double call_tp(double x) {
  return sin(x);
}
// CHECK-LABEL: define {{.*}} double @call_tp(
// CHECK:       call double @sin_rtp(

#pragma STDC FENV_ROUND FE_TONEARESTFROMZERO
double call_tea(double x) {
  return sin(x);
}
// CHECK-LABEL: define {{.*}} double @call_tea(
// CHECK:       call double @sin_rta(
