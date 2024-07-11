// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=basic -o - | FileCheck --check-prefix=BASIC %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fno-cx-limited-range -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=improved -o - | FileCheck --check-prefix=IMPRVD %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=promoted -o - | FileCheck --check-prefix=PRMTD %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=full -o - | FileCheck --check-prefix=FULL %s

_Complex float pragma_on_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE ON
  // LABEL: define {{.*}} @pragma_on_mul(

  // FULL: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fsub float
  // FULL-NEXT: fadd float

  // BASIC: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fsub float
  // BASIC-NEXT: fadd float

  // IMPRVD: fmul float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fsub float
  // IMPRVD-NEXT: fadd float

  // PRMTD: fmul float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fsub float
  // PRMTD-NEXT: fadd float

  return a * b;
}

_Complex float pragma_off_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // LABEL: define {{.*}} @pragma_off_mul(

  // FULL: call {{.*}} @__mulsc3

  // BASIC: call {{.*}} @__mulsc3

  // IMPRVD: call {{.*}} @__mulsc3

  // PRMTD: call {{.*}} @__mulsc3

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

  // BASIC: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fadd float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fadd float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fsub float
  // BASIC-NEXT: fdiv float
  // BASIC-NEXT: fdiv float

  // IMPRVD: fmul float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fadd float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fadd float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fsub float
  // IMPRVD-NEXT: fdiv float
  // IMPRVD-NEXT: fdiv float

  // PRMTD: fmul float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fadd float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fadd float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fsub float
  // PRMTD-NEXT: fdiv float
  // PRMTD-NEXT: fdiv float

  return a / b;
}

_Complex float pragma_off_div(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // LABEL: define {{.*}} @pragma_off_div(

  // FULL: call {{.*}} @__divsc3

  // BASIC: call {{.*}} @__divsc3

  // IMPRVD: call {{.*}} @__divsc3

  // PRMTD: call {{.*}} @__divsc3

  return a / b;
}

_Complex float pragma_default_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE DEFAULT
  // LABEL: define {{.*}} @pragma_on_mul(

  // FULL: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fmul float
  // FULL-NEXT: fsub float
  // FULL-NEXT: fadd float

  // BASIC: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fsub float
  // BASIC-NEXT: fadd float

  // IMPRVD: fmul float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fsub float
  // IMPRVD-NEXT: fadd float

  // PRMTD: fmul float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fsub float
  // PRMTD-NEXT: fadd float

  return a * b;
}
_Complex float pragma_default_div(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE DEFAULT
  // LABEL: define {{.*}} @pragma_on_divx(

  // FULL: call {{.*}} @__divsc3

  // BASIC: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fadd float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fadd float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fmul float
  // BASIC-NEXT: fsub float
  // BASIC-NEXT: fdiv float
  // BASIC-NEXT: fdiv float

  // IMPRVD: call{{.*}}float @llvm.fabs.f32(float {{.*}})
  // IMPRVD-NEXT: call{{.*}}float @llvm.fabs.f32(float {{.*}})
  // IMPRVD-NEXT: fcmp{{.*}}ugt float {{.*}}, {{.*}}
  // IMPRVD-NEXT:   br i1 {{.*}}, label
  // IMPRVD:  abs_rhsr_greater_or_equal_abs_rhsi:
  // IMPRVD-NEXT: fdiv float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fadd float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fadd float
  // IMPRVD-NEXT: fdiv float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fsub float
  // IMPRVD-NEXT: fdiv float
  // IMPRVD-NEXT: br label
  // IMPRVD: abs_rhsr_less_than_abs_rhsi:
  // IMPRVD-NEXT: fdiv float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fadd float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fadd float
  // IMPRVD-NEXT: fdiv float
  // IMPRVD-NEXT: fmul float
  // IMPRVD-NEXT: fsub float
  // IMPRVD-NEXT: fdiv float

  // PRMTD: call{{.*}}float @llvm.fabs.f32(float {{.*}})
  // PRMTD-NEXT: call{{.*}}float @llvm.fabs.f32(float {{.*}})
  // PRMTD-NEXT: fcmp{{.*}}ugt float {{.*}}, {{.*}}
  // PRMTD-NEXT:   br i1 {{.*}}, label
  // PRMTD:  abs_rhsr_greater_or_equal_abs_rhsi:
  // PRMTD-NEXT: fdiv float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fadd float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fadd float
  // PRMTD-NEXT: fdiv float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fsub float
  // PRMTD-NEXT: fdiv float
  // PRMTD-NEXT: br label
  // PRMTD: abs_rhsr_less_than_abs_rhsi:
  // PRMTD-NEXT: fdiv float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fadd float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fadd float
  // PRMTD-NEXT: fdiv float
  // PRMTD-NEXT: fmul float
  // PRMTD-NEXT: fsub float
  // PRMTD-NEXT: fdiv float

  return a / b;
}
