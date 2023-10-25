// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fcx-limited-range -o - | FileCheck --check-prefix=LMTD %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fno-cx-limited-range -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fcx-fortran-rules -o - | FileCheck --check-prefix=FRTRN %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fno-cx-fortran-rules -o - | FileCheck --check-prefix=FULL %s

_Complex float pragma_on_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE ON
  // LABEL: define {{.*}} @pragma_on_mul(
  // FULL: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fsub float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fadd float

  // LMTD: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fsub float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fadd float

  // FRTRN: fmul float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fsub float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fadd float

  return a * b;
}

_Complex float pragma_off_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // LABEL: define {{.*}} @pragma_off_mul(
  // FULL: call {{.*}} @__mulsc3

  // LMTD: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fsub float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fadd float

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

  // FRTRN: call float @llvm.fabs.f32(float {{.*}})
  // FRTRN-NEXT: call float @llvm.fabs.f32(float {{.*}})
  // FRTRN-NEXT: fcmp ugt float
  // FRTRN-NEXT: br i1 {{.*}}, label
  // FRTRN:      true_bb_name:
  // FRTRN-NEXT: fdiv float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fadd float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fadd float
  // FRTRN-NEXT: fdiv float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fsub float
  // FRTRN-NEXT: fdiv float
  // FRTRN-NEXT: br label
  // FRTRN:      false_bb_name:
  // FRTRN-NEXT: fdiv float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fadd float
  // FRTRN-NEXT: fadd float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fdiv float
  // FRTRN-NEXT: fsub float
  // FRTRN-NEXT: fmul float
  // FRTRN-NEXT: fdiv float
  // FRTRN-NEXT: br label
  // FRTRN:      cont_bb:
  // FRTRN-NEXT: phi float
  // FRTRN-NEXT: phi float

  return a / b;
}
