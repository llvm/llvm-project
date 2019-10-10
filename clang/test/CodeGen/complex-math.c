// FIXME: This file should not be using -O1; that makes it depend on the entire LLVM IR optimizer.

// RUN: %clang_cc1 %s -O1 -fno-experimental-new-pass-manager -emit-llvm -triple x86_64-unknown-unknown -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -O1 -fno-experimental-new-pass-manager -emit-llvm -triple x86_64-pc-win64 -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -O1 -fno-experimental-new-pass-manager -emit-llvm -triple i686-unknown-unknown -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -O1 -fno-experimental-new-pass-manager -emit-llvm -triple powerpc-unknown-unknown -o - | FileCheck %s --check-prefix=PPC
// RUN: %clang_cc1 %s -O1 -fno-experimental-new-pass-manager -emit-llvm -triple armv7-none-linux-gnueabi -o - | FileCheck %s --check-prefix=ARM
// RUN: %clang_cc1 %s -O1 -fno-experimental-new-pass-manager -emit-llvm -triple armv7-none-linux-gnueabihf -o - | FileCheck %s --check-prefix=ARMHF
// RUN: %clang_cc1 %s -O1 -fno-experimental-new-pass-manager -emit-llvm -triple thumbv7k-apple-watchos2.0 -o - -target-abi aapcs16 | FileCheck %s --check-prefix=ARM7K
// RUN: %clang_cc1 %s -O1 -fno-experimental-new-pass-manager -emit-llvm -triple aarch64-unknown-unknown -ffast-math -o - | FileCheck %s --check-prefix=AARCH64-FASTMATH

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
  // X86: fneg
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

  // a / b = (A+iB) / (C+iD) = ((AC+BD)/(CC+DD)) + i((BC-AD)/(CC+DD))
  // AARCH64-FASTMATH-LABEL: @div_float_rc(float %a, [2 x float] %b.coerce)
  // A = a
  // B = 0
  // AARCH64-FASTMATH: [[C:%.*]] = extractvalue [2 x float] %b.coerce, 0
  // AARCH64-FASTMATH: [[D:%.*]] = extractvalue [2 x float] %b.coerce, 1
  //
  // AARCH64-FASTMATH: [[AC:%.*]] = fmul fast float [[C]], %a
  // BD = 0
  // ACpBD = AC
  //
  // AARCH64-FASTMATH: [[CC:%.*]] = fmul fast float [[C]], [[C]]
  // AARCH64-FASTMATH: [[DD:%.*]] = fmul fast float [[D]], [[D]]
  // AARCH64-FASTMATH: [[CCpDD:%.*]] = fadd fast float [[CC]], [[DD]]
  //
  // BC = 0
  // AARCH64-FASTMATH: [[NEGA:%.*]] = fsub fast float -0.000000e+00, %a
  // AARCH64-FASTMATH: [[AD:%.*]] = fmul fast float  [[D]], [[NEGA]]
  //
  // AARCH64-FASTMATH: fdiv fast float [[AC]], [[CCpDD]]
  // AARCH64-FASTMATH: fdiv fast float [[AD]], [[CCpDD]]
  // AARCH64-FASTMATH: ret
  return a / b;
}
float _Complex div_float_cc(float _Complex a, float _Complex b) {
  // X86-LABEL: @div_float_cc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divsc3(
  // X86: ret

  // a / b = (A+iB) / (C+iD) = ((AC+BD)/(CC+DD)) + i((BC-AD)/(CC+DD))
  // AARCH64-FASTMATH-LABEL: @div_float_cc([2 x float] %a.coerce, [2 x float] %b.coerce)
  // AARCH64-FASTMATH: [[A:%.*]] = extractvalue [2 x float] %a.coerce, 0
  // AARCH64-FASTMATH: [[B:%.*]] = extractvalue [2 x float] %a.coerce, 1
  // AARCH64-FASTMATH: [[C:%.*]] = extractvalue [2 x float] %b.coerce, 0
  // AARCH64-FASTMATH: [[D:%.*]] = extractvalue [2 x float] %b.coerce, 1
  //
  // AARCH64-FASTMATH: [[AC:%.*]] = fmul fast float [[C]], [[A]]
  // AARCH64-FASTMATH: [[BD:%.*]] = fmul fast float [[D]], [[B]]
  // AARCH64-FASTMATH: [[ACpBD:%.*]] = fadd fast float [[AC]], [[BD]]
  //
  // AARCH64-FASTMATH: [[CC:%.*]] = fmul fast float [[C]], [[C]]
  // AARCH64-FASTMATH: [[DD:%.*]] = fmul fast float [[D]], [[D]]
  // AARCH64-FASTMATH: [[CCpDD:%.*]] = fadd fast float [[CC]], [[DD]]
  //
  // AARCH64-FASTMATH: [[BC:%.*]] = fmul fast float [[C]], [[B]]
  // AARCH64-FASTMATH: [[AD:%.*]] = fmul fast float [[D]], [[A]]
  // AARCH64-FASTMATH: [[BCmAD:%.*]] = fsub fast float [[BC]], [[AD]]
  //
  // AARCH64-FASTMATH: fdiv fast float [[ACpBD]], [[CCpDD]]
  // AARCH64-FASTMATH: fdiv fast float [[BCmAD]], [[CCpDD]]
  // AARCH64-FASTMATH: ret
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
  // X86: fneg
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

  // a / b = (A+iB) / (C+iD) = ((AC+BD)/(CC+DD)) + i((BC-AD)/(CC+DD))
  // AARCH64-FASTMATH-LABEL: @div_double_rc(double %a, [2 x double] %b.coerce)
  // A = a
  // B = 0
  // AARCH64-FASTMATH: [[C:%.*]] = extractvalue [2 x double] %b.coerce, 0
  // AARCH64-FASTMATH: [[D:%.*]] = extractvalue [2 x double] %b.coerce, 1
  //
  // AARCH64-FASTMATH: [[AC:%.*]] = fmul fast double [[C]], %a
  // BD = 0
  // ACpBD = AC
  //
  // AARCH64-FASTMATH: [[CC:%.*]] = fmul fast double [[C]], [[C]]
  // AARCH64-FASTMATH: [[DD:%.*]] = fmul fast double [[D]], [[D]]
  // AARCH64-FASTMATH: [[CCpDD:%.*]] = fadd fast double [[CC]], [[DD]]
  //
  // BC = 0
  // AARCH64-FASTMATH: [[NEGA:%.*]] = fsub fast double -0.000000e+00, %a
  // AARCH64-FASTMATH: [[AD:%.*]] = fmul fast double [[D]], [[NEGA]]
  //
  // AARCH64-FASTMATH: fdiv fast double [[AC]], [[CCpDD]]
  // AARCH64-FASTMATH: fdiv fast double [[AD]], [[CCpDD]]
  // AARCH64-FASTMATH: ret
  return a / b;
}
double _Complex div_double_cc(double _Complex a, double _Complex b) {
  // X86-LABEL: @div_double_cc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divdc3(
  // X86: ret

  // a / b = (A+iB) / (C+iD) = ((AC+BD)/(CC+DD)) + i((BC-AD)/(CC+DD))
  // AARCH64-FASTMATH-LABEL: @div_double_cc([2 x double] %a.coerce, [2 x double] %b.coerce)
  // AARCH64-FASTMATH: [[A:%.*]] = extractvalue [2 x double] %a.coerce, 0
  // AARCH64-FASTMATH: [[B:%.*]] = extractvalue [2 x double] %a.coerce, 1
  // AARCH64-FASTMATH: [[C:%.*]] = extractvalue [2 x double] %b.coerce, 0
  // AARCH64-FASTMATH: [[D:%.*]] = extractvalue [2 x double] %b.coerce, 1
  //
  // AARCH64-FASTMATH: [[AC:%.*]] = fmul fast double [[C]], [[A]]
  // AARCH64-FASTMATH: [[BD:%.*]] = fmul fast double [[D]], [[B]]
  // AARCH64-FASTMATH: [[ACpBD:%.*]] = fadd fast double [[AC]], [[BD]]
  //
  // AARCH64-FASTMATH: [[CC:%.*]] = fmul fast double [[C]], [[C]]
  // AARCH64-FASTMATH: [[DD:%.*]] = fmul fast double [[D]], [[D]]
  // AARCH64-FASTMATH: [[CCpDD:%.*]] = fadd fast double [[CC]], [[DD]]
  //
  // AARCH64-FASTMATH: [[BC:%.*]] = fmul fast double [[C]], [[B]]
  // AARCH64-FASTMATH: [[AD:%.*]] = fmul fast double [[D]], [[A]]
  // AARCH64-FASTMATH: [[BCmAD:%.*]] = fsub fast double [[BC]], [[AD]]
  //
  // AARCH64-FASTMATH: fdiv fast double [[ACpBD]], [[CCpDD]]
  // AARCH64-FASTMATH: fdiv fast double [[BCmAD]], [[CCpDD]]
  // AARCH64-FASTMATH: ret
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
  // X86: fneg
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

  // a / b = (A+iB) / (C+iD) = ((AC+BD)/(CC+DD)) + i((BC-AD)/(CC+DD))
  // AARCH64-FASTMATH-LABEL: @div_long_double_rc(fp128 %a, [2 x fp128] %b.coerce)
  // A = a
  // B = 0
  // AARCH64-FASTMATH: [[C:%.*]] = extractvalue [2 x fp128] %b.coerce, 0
  // AARCH64-FASTMATH: [[D:%.*]] = extractvalue [2 x fp128] %b.coerce, 1
  //
  // AARCH64-FASTMATH: [[AC:%.*]] = fmul fast fp128 [[C]], %a
  // BD = 0
  // ACpBD = AC
  //
  // AARCH64-FASTMATH: [[CC:%.*]] = fmul fast fp128 [[C]], [[C]]
  // AARCH64-FASTMATH: [[DD:%.*]] = fmul fast fp128 [[D]], [[D]]
  // AARCH64-FASTMATH: [[CCpDD:%.*]] = fadd fast fp128 [[CC]], [[DD]]
  //
  // BC = 0
  // AARCH64-FASTMATH: [[NEGA:%.*]] = fsub fast fp128 0xL00000000000000008000000000000000, %a
  // AARCH64-FASTMATH: [[AD:%.*]] = fmul fast fp128 [[D]], [[NEGA]]
  //
  // AARCH64-FASTMATH: fdiv fast fp128 [[AC]], [[CCpDD]]
  // AARCH64-FASTMATH: fdiv fast fp128 [[AD]], [[CCpDD]]
  // AARCH64-FASTMATH: ret
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

  // a / b = (A+iB) / (C+iD) = ((AC+BD)/(CC+DD)) + i((BC-AD)/(CC+DD))
  // AARCH64-FASTMATH-LABEL: @div_long_double_cc([2 x fp128] %a.coerce, [2 x fp128] %b.coerce)
  // AARCH64-FASTMATH: [[A:%.*]] = extractvalue [2 x fp128] %a.coerce, 0
  // AARCH64-FASTMATH: [[B:%.*]] = extractvalue [2 x fp128] %a.coerce, 1
  // AARCH64-FASTMATH: [[C:%.*]] = extractvalue [2 x fp128] %b.coerce, 0
  // AARCH64-FASTMATH: [[D:%.*]] = extractvalue [2 x fp128] %b.coerce, 1
  //
  // AARCH64-FASTMATH: [[AC:%.*]] = fmul fast fp128 [[C]], [[A]]
  // AARCH64-FASTMATH: [[BD:%.*]] = fmul fast fp128 [[D]], [[B]]
  // AARCH64-FASTMATH: [[ACpBD:%.*]] = fadd fast fp128 [[AC]], [[BD]]
  //
  // AARCH64-FASTMATH: [[CC:%.*]] = fmul fast fp128 [[C]], [[C]]
  // AARCH64-FASTMATH: [[DD:%.*]] = fmul fast fp128 [[D]], [[D]]
  // AARCH64-FASTMATH: [[CCpDD:%.*]] = fadd fast fp128 [[CC]], [[DD]]
  //
  // AARCH64-FASTMATH: [[BC:%.*]] = fmul fast fp128 [[C]], [[B]]
  // AARCH64-FASTMATH: [[AD:%.*]] = fmul fast fp128 [[D]], [[A]]
  // AARCH64-FASTMATH: [[BCmAD:%.*]] = fsub fast fp128 [[BC]], [[AD]]
  //
  // AARCH64-FASTMATH: fdiv fast fp128 [[ACpBD]], [[CCpDD]]
  // AARCH64-FASTMATH: fdiv fast fp128 [[BCmAD]], [[CCpDD]]
  // AARCH64-FASTMATH: ret
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
  // These functions are not defined as floating point helper functions in
  // Run-time ABI for the ARM architecture document so they must not always
  // use the base AAPCS.

  // ARM-LABEL: @foo(
  // ARM: call void @__muldc3

  // ARMHF-LABEL: @foo(
  // ARMHF: call { double, double } @__muldc3

  // ARM7K-LABEL: @foo(
  // ARM7K: call { double, double } @__muldc3
  return a*b;
}
