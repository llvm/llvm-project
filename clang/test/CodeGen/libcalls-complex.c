// RUN: %clang_cc1 -fno-builtin -emit-llvm -o - %s -triple i386-unknown-unknown | FileCheck -check-prefix CHECK-YES %s
// RUN: %clang_cc1 -emit-llvm -o - %s -triple i386-unknown-unknown | FileCheck -check-prefix CHECK-NO %s

extern float crealf(float _Complex);
extern double creal(double _Complex);
extern long double creall(long double _Complex);

extern float cimagf(float _Complex);
extern double cimag(double _Complex);
extern long double cimagl(long double _Complex);

double test_creal(double _Complex z) {
  return creal(z);
  // CHECK-NO-NOT: call double @creal
  // CHECK-YES: call double @creal
}

long double test_creall(double _Complex z) {
  return creall(z);
  // CHECK-NO-NOT: call x86_fp80 @creall
  // CHECK-YES: call x86_fp80 @creall
}

float test_crealf(double _Complex z) {
  return crealf(z);
  // CHECK-NO-NOT: call float @crealf
  // CHECK-YES: call float @crealf
}

double test_cimag(double _Complex z) {
  return cimag(z);
  // CHECK-NO-NOT: call double @cimag
  // CHECK-YES: call double @cimag
}

long double test_cimagl(double _Complex z) {
  return cimagl(z);
  // CHECK-NO-NOT: call x86_fp80 @cimagl
  // CHECK-YES: call x86_fp80 @cimagl
}

float test_cimagf(double _Complex z) {
  return cimagf(z);
  // CHECK-NO-NOT: call float @cimagf
  // CHECK-YES: call float @cimagf
}
