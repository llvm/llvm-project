// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=limited -o - | FileCheck %s --check-prefix=LMTD

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fno-cx-limited-range -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=fortran -o - | FileCheck %s --check-prefix=FRTRN

// Fast math
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN: -ffast-math -complex-range=limited -emit-llvm -o - %s \
// RUN: | FileCheck %s --check-prefix=LMTD-FAST

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fno-cx-fortran-rules -o - | FileCheck %s --check-prefix=FULL

_Complex float div(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @div(
  // FULL:  call {{.*}} @__divsc3

  // LMTD:      fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fadd float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fadd float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fsub float
  // LMTD-NEXT: fdiv float
  // LMTD-NEXT: fdiv float

  // FRTRN: call {{.*}}float @llvm.fabs.f32(float {{.*}})
  // FRTRN-NEXT: call {{.*}}float @llvm.fabs.f32(float {{.*}})
  // FRTRN-NEXT: fcmp {{.*}}ugt float
  // FRTRN-NEXT: br i1 {{.*}}, label
  // FRTRN:      abs_rhsr_greater_or_equal_abs_rhsi:
  // FRTRN-NEXT: fdiv {{.*}}float
  // FRTRN-NEXT: fmul {{.*}}float
  // FRTRN-NEXT: fadd {{.*}}float
  // FRTRN-NEXT: fmul {{.*}}float
  // FRTRN-NEXT: fadd {{.*}}float
  // FRTRN-NEXT: fdiv {{.*}}float
  // FRTRN-NEXT: fmul {{.*}}float
  // FRTRN-NEXT: fsub {{.*}}float
  // FRTRN-NEXT: fdiv {{.*}}float
  // FRTRN-NEXT: br label
  // FRTRN:      abs_rhsr_less_than_abs_rhsi:
  // FRTRN-NEXT: fdiv {{.*}}float
  // FRTRN-NEXT: fmul {{.*}}float
  // FRTRN-NEXT: fadd {{.*}}float
  // FRTRN-NEXT: fmul {{.*}}float
  // FRTRN-NEXT: fadd {{.*}}float
  // FRTRN-NEXT: fdiv {{.*}}float
  // FRTRN-NEXT: fmul {{.*}}float
  // FRTRN-NEXT: fsub {{.*}}float
  // FRTRN-NEXT: fdiv {{.*}}float
  // FRTRN-NEXT: br label
  // FRTRN:      complex_div:
  // FRTRN-NEXT: phi {{.*}}float
  // FRTRN-NEXT: phi {{.*}}float

  // LMTD-FAST: fmul {{.*}} float
  // LMTD-FAST-NEXT: fmul {{.*}} float
  // LMTD-FAST-NEXT: fadd {{.*}} float
  // LMTD-FAST-NEXT: fmul {{.*}} float
  // LMTD-FAST-NEXT: fmul {{.*}} float
  // LMTD-FAST-NEXT: fadd {{.*}} float
  // LMTD-FAST-NEXT: fmul {{.*}} float
  // LMTD-FAST-NEXT: fmul {{.*}} float
  // LMTD-FAST-NEXT: fsub {{.*}} float
  // LMTD-FAST-NEXT: fdiv {{.*}} float
  // LMTD-FAST-NEXT: fdiv {{.*}} float

  return a / b;
}

_Complex float mul(_Complex float a, _Complex float b) {
  // LABEL: define {{.*}} @mul(
  // FULL:  call {{.*}} @__mulsc3

  // LMTD: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fsub float
  // LMTD-NEXT: fadd float

  // FRTRN: fmul {{.*}}float
  // FRTRN-NEXT: fmul {{.*}}float
  // FRTRN-NEXT: fmul {{.*}}float
  // FRTRN-NEXT: fmul {{.*}}float
  // FRTRN-NEXT: fsub {{.*}}float
  // FRTRN-NEXT: fadd {{.*}}float

  // LMTD-FAST: fmul {{.*}} float
  // LMTD-FAST-NEXT: fmul {{.*}} float
  // LMTD-FAST-NEXT: fmul {{.*}} float
  // LMTD-FAST-NEXT: fmul {{.*}} float
  // LMTD-FAST-NEXT: fsub {{.*}} float
  // LMTD-FAST-NEXT: fadd {{.*}} float

  return a * b;
}
