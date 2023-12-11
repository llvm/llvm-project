// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=limited -o - | FileCheck --check-prefix=LMTD %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fno-cx-limited-range -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=fortran -o - | FileCheck --check-prefix=FRTRN %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fno-cx-fortran-rules -o - | FileCheck --check-prefix=FULL %s

_Complex float pragma_on_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE ON
  // LABEL: define {{.*}} @pragma_on_mul(
  // FULL: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fsub float
  // FULL-NEXT: fadd float

  // LMTD: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fsub float
  // LMTD-NEXT: fadd float

  // FRTRN: fmul float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fsub float
  // FRTRN-NEXT: fadd float

  return a * b;
}

_Complex float pragma_off_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // LABEL: define {{.*}} @pragma_off_mul(
  // FULL: call {{.*}} @__mulsc3

  // LMTD: call {{.*}} @__mulsc3

  // FRTRN: call {{.*}} @__mulsc3

  return a * b;
}

_Complex float pragma_on_div(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE ON
  // LABEL: define {{.*}} @pragma_on_div(
  // FULL: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fadd float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fadd float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fsub float
  // FULL-NEXT: fdiv float
  // FULL: fdiv float

  // LMTD: fmul float
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

  // FRTRN: fmul float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fadd float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fadd float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fsub float
  // FRTRN-NEXT: fdiv float
  // FRTRN-NEXT: fdiv float

  return a / b;
}

_Complex float pragma_off_div(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // LABEL: define {{.*}} @pragma_off_div(
  // FULL: call {{.*}} @__divsc3

  // LMTD: call {{.*}} @__divsc3

  // FRTRN: call {{.*}} @__divsc3

  return a / b;
}
