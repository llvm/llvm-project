// RUN: %clang_cc1 %s -O1 -emit-llvm -triple x86_64-unknown-unknown -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -O1 -emit-llvm -triple x86_64-pc-win64 -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -O1 -emit-llvm -triple i686-unknown-unknown -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -O1 -emit-llvm -triple powerpc-unknown-unknown -o - | FileCheck %s --check-prefix=PPC
// RUN: %clang_cc1 %s -O1 -emit-llvm -triple armv7-none-linux-gnueabihf -o - | FileCheck %s --check-prefix=ARM
// RUN: %clang_cc1 %s -O1 -emit-llvm -triple thumbv7k-apple-watchos2.0 -o - -target-abi aapcs16 | FileCheck %s --check-prefix=ARM7K

float _Complex add_float_rr(float a, float b) {
  // X86-LABEL: @add_float_rr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
float _Complex add_float_cr(float _Complex a, float b) {
  // X86-LABEL: @add_float_cr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
float _Complex add_float_rc(float a, float _Complex b) {
  // X86-LABEL: @add_float_rc(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
float _Complex add_float_cc(float _Complex a, float _Complex b) {
  // X86-LABEL: @add_float_cc(
  // X86: fadd
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}

float _Complex sub_float_rr(float a, float b) {
  // X86-LABEL: @sub_float_rr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
float _Complex sub_float_cr(float _Complex a, float b) {
  // X86-LABEL: @sub_float_cr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
float _Complex sub_float_rc(float a, float _Complex b) {
  // X86-LABEL: @sub_float_rc(
  // X86: fsub
  // X86: fsub float -0.{{0+}}e+00,
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
float _Complex sub_float_cc(float _Complex a, float _Complex b) {
  // X86-LABEL: @sub_float_cc(
  // X86: fsub
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}

float _Complex mul_float_rr(float a, float b) {
  // X86-LABEL: @mul_float_rr(
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
float _Complex mul_float_cr(float _Complex a, float b) {
  // X86-LABEL: @mul_float_cr(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
float _Complex mul_float_rc(float a, float _Complex b) {
  // X86-LABEL: @mul_float_rc(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
float _Complex mul_float_cc(float _Complex a, float _Complex b) {
  // X86-LABEL: @mul_float_cc(
  // X86: %[[AC:[^ ]+]] = fmul
  // X86: %[[BD:[^ ]+]] = fmul
  // X86: %[[AD:[^ ]+]] = fmul
  // X86: %[[BC:[^ ]+]] = fmul
  // X86: %[[RR:[^ ]+]] = fsub float %[[AC]], %[[BD]]
  // X86: %[[RI:[^ ]+]] = fadd float
  // X86-DAG: %[[AD]]
  // X86-DAG: ,
  // X86-DAG: %[[BC]]
  // X86: fcmp uno float %[[RR]]
  // X86: fcmp uno float %[[RI]]
  // X86: call {{.*}} @__mulsc3(
  // X86: ret
  return a * b;
}

float _Complex div_float_rr(float a, float b) {
  // X86-LABEL: @div_float_rr(
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
float _Complex div_float_cr(float _Complex a, float b) {
  // X86-LABEL: @div_float_cr(
  // X86: fdiv
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
float _Complex div_float_rc(float a, float _Complex b) {
  // X86-LABEL: @div_float_rc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divsc3(
  // X86: ret
  return a / b;
}
float _Complex div_float_cc(float _Complex a, float _Complex b) {
  // X86-LABEL: @div_float_cc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divsc3(
  // X86: ret
  return a / b;
}

double _Complex add_double_rr(double a, double b) {
  // X86-LABEL: @add_double_rr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
double _Complex add_double_cr(double _Complex a, double b) {
  // X86-LABEL: @add_double_cr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
double _Complex add_double_rc(double a, double _Complex b) {
  // X86-LABEL: @add_double_rc(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
double _Complex add_double_cc(double _Complex a, double _Complex b) {
  // X86-LABEL: @add_double_cc(
  // X86: fadd
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}

double _Complex sub_double_rr(double a, double b) {
  // X86-LABEL: @sub_double_rr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
double _Complex sub_double_cr(double _Complex a, double b) {
  // X86-LABEL: @sub_double_cr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
double _Complex sub_double_rc(double a, double _Complex b) {
  // X86-LABEL: @sub_double_rc(
  // X86: fsub
  // X86: fsub double -0.{{0+}}e+00,
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
double _Complex sub_double_cc(double _Complex a, double _Complex b) {
  // X86-LABEL: @sub_double_cc(
  // X86: fsub
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}

double _Complex mul_double_rr(double a, double b) {
  // X86-LABEL: @mul_double_rr(
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
double _Complex mul_double_cr(double _Complex a, double b) {
  // X86-LABEL: @mul_double_cr(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
double _Complex mul_double_rc(double a, double _Complex b) {
  // X86-LABEL: @mul_double_rc(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
double _Complex mul_double_cc(double _Complex a, double _Complex b) {
  // X86-LABEL: @mul_double_cc(
  // X86: %[[AC:[^ ]+]] = fmul
  // X86: %[[BD:[^ ]+]] = fmul
  // X86: %[[AD:[^ ]+]] = fmul
  // X86: %[[BC:[^ ]+]] = fmul
  // X86: %[[RR:[^ ]+]] = fsub double %[[AC]], %[[BD]]
  // X86: %[[RI:[^ ]+]] = fadd double
  // X86-DAG: %[[AD]]
  // X86-DAG: ,
  // X86-DAG: %[[BC]]
  // X86: fcmp uno double %[[RR]]
  // X86: fcmp uno double %[[RI]]
  // X86: call {{.*}} @__muldc3(
  // X86: ret
  return a * b;
}

double _Complex div_double_rr(double a, double b) {
  // X86-LABEL: @div_double_rr(
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
double _Complex div_double_cr(double _Complex a, double b) {
  // X86-LABEL: @div_double_cr(
  // X86: fdiv
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
double _Complex div_double_rc(double a, double _Complex b) {
  // X86-LABEL: @div_double_rc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divdc3(
  // X86: ret
  return a / b;
}
double _Complex div_double_cc(double _Complex a, double _Complex b) {
  // X86-LABEL: @div_double_cc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divdc3(
  // X86: ret
  return a / b;
}

long double _Complex add_long_double_rr(long double a, long double b) {
  // X86-LABEL: @add_long_double_rr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
long double _Complex add_long_double_cr(long double _Complex a, long double b) {
  // X86-LABEL: @add_long_double_cr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
long double _Complex add_long_double_rc(long double a, long double _Complex b) {
  // X86-LABEL: @add_long_double_rc(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
long double _Complex add_long_double_cc(long double _Complex a, long double _Complex b) {
  // X86-LABEL: @add_long_double_cc(
  // X86: fadd
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}

long double _Complex sub_long_double_rr(long double a, long double b) {
  // X86-LABEL: @sub_long_double_rr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
long double _Complex sub_long_double_cr(long double _Complex a, long double b) {
  // X86-LABEL: @sub_long_double_cr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
long double _Complex sub_long_double_rc(long double a, long double _Complex b) {
  // X86-LABEL: @sub_long_double_rc(
  // X86: fsub
  // X86: fsub x86_fp80 0xK8{{0+}},
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
long double _Complex sub_long_double_cc(long double _Complex a, long double _Complex b) {
  // X86-LABEL: @sub_long_double_cc(
  // X86: fsub
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}

long double _Complex mul_long_double_rr(long double a, long double b) {
  // X86-LABEL: @mul_long_double_rr(
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
long double _Complex mul_long_double_cr(long double _Complex a, long double b) {
  // X86-LABEL: @mul_long_double_cr(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
long double _Complex mul_long_double_rc(long double a, long double _Complex b) {
  // X86-LABEL: @mul_long_double_rc(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
long double _Complex mul_long_double_cc(long double _Complex a, long double _Complex b) {
  // X86-LABEL: @mul_long_double_cc(
  // X86: %[[AC:[^ ]+]] = fmul
  // X86: %[[BD:[^ ]+]] = fmul
  // X86: %[[AD:[^ ]+]] = fmul
  // X86: %[[BC:[^ ]+]] = fmul
  // X86: %[[RR:[^ ]+]] = fsub x86_fp80 %[[AC]], %[[BD]]
  // X86: %[[RI:[^ ]+]] = fadd x86_fp80
  // X86-DAG: %[[AD]]
  // X86-DAG: ,
  // X86-DAG: %[[BC]]
  // X86: fcmp uno x86_fp80 %[[RR]]
  // X86: fcmp uno x86_fp80 %[[RI]]
  // X86: call {{.*}} @__mulxc3(
  // X86: ret
  // PPC-LABEL: @mul_long_double_cc(
  // PPC: %[[AC:[^ ]+]] = fmul
  // PPC: %[[BD:[^ ]+]] = fmul
  // PPC: %[[AD:[^ ]+]] = fmul
  // PPC: %[[BC:[^ ]+]] = fmul
  // PPC: %[[RR:[^ ]+]] = fsub ppc_fp128 %[[AC]], %[[BD]]
  // PPC: %[[RI:[^ ]+]] = fadd ppc_fp128
  // PPC-DAG: %[[AD]]
  // PPC-DAG: ,
  // PPC-DAG: %[[BC]]
  // PPC: fcmp uno ppc_fp128 %[[RR]]
  // PPC: fcmp uno ppc_fp128 %[[RI]]
  // PPC: call {{.*}} @__multc3(
  // PPC: ret
  return a * b;
}

long double _Complex div_long_double_rr(long double a, long double b) {
  // X86-LABEL: @div_long_double_rr(
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
long double _Complex div_long_double_cr(long double _Complex a, long double b) {
  // X86-LABEL: @div_long_double_cr(
  // X86: fdiv
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
long double _Complex div_long_double_rc(long double a, long double _Complex b) {
  // X86-LABEL: @div_long_double_rc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divxc3(
  // X86: ret
  // PPC-LABEL: @div_long_double_rc(
  // PPC-NOT: fdiv
  // PPC: call {{.*}} @__divtc3(
  // PPC: ret
  return a / b;
}
long double _Complex div_long_double_cc(long double _Complex a, long double _Complex b) {
  // X86-LABEL: @div_long_double_cc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divxc3(
  // X86: ret
  // PPC-LABEL: @div_long_double_cc(
  // PPC-NOT: fdiv
  // PPC: call {{.*}} @__divtc3(
  // PPC: ret
  return a / b;
}

// Comparison operators don't rely on library calls or have interseting math
// properties, but test that mixed types work correctly here.
_Bool eq_float_cr(float _Complex a, float b) {
  // X86-LABEL: @eq_float_cr(
  // X86: fcmp oeq
  // X86: fcmp oeq
  // X86: and i1
  // X86: ret
  return a == b;
}
_Bool eq_float_rc(float a, float _Complex b) {
  // X86-LABEL: @eq_float_rc(
  // X86: fcmp oeq
  // X86: fcmp oeq
  // X86: and i1
  // X86: ret
  return a == b;
}
_Bool eq_float_cc(float _Complex a, float _Complex b) {
  // X86-LABEL: @eq_float_cc(
  // X86: fcmp oeq
  // X86: fcmp oeq
  // X86: and i1
  // X86: ret
  return a == b;
}
_Bool ne_float_cr(float _Complex a, float b) {
  // X86-LABEL: @ne_float_cr(
  // X86: fcmp une
  // X86: fcmp une
  // X86: or i1
  // X86: ret
  return a != b;
}
_Bool ne_float_rc(float a, float _Complex b) {
  // X86-LABEL: @ne_float_rc(
  // X86: fcmp une
  // X86: fcmp une
  // X86: or i1
  // X86: ret
  return a != b;
}
_Bool ne_float_cc(float _Complex a, float _Complex b) {
  // X86-LABEL: @ne_float_cc(
  // X86: fcmp une
  // X86: fcmp une
  // X86: or i1
  // X86: ret
  return a != b;
}

// Check that the libcall will obtain proper calling convention on ARM
_Complex double foo(_Complex double a, _Complex double b) {
  // ARM-LABEL: @foo(
  // ARM: call arm_aapcscc { double, double } @__muldc3

  // ARM7K-LABEL: @foo(
  // ARM7K: call { double, double } @__muldc3
  return a*b;
}
