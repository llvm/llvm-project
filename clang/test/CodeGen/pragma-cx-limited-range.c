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

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -ffast-math -ffp-contract=fast -complex-range=basic -o - \
// RUN: | FileCheck --check-prefix=FAST %s

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

  // PRMTD: load float, ptr {{.*}}
  // PRMTD: fpext float {{.*}} to double
  // PRMTD-NEXT: fpext float {{.*}} to double
  // PRMTD-NEXT: getelementptr inbounds nuw { float, float }, ptr {{.*}}, i32 0, i32 0
  // PRMTD-NEXT: load float, ptr {{.*}}
  // PRMTD-NEXT: getelementptr inbounds nuw { float, float }, ptr {{.*}}, i32 0, i32 1
  // PRMTD-NEXT: load float, ptr {{.*}}
  // PRMTD-NEXT: fpext float {{.*}} to double
  // PRMTD-NEXT: fpext float {{.*}} to double
  // PRMTD-NEXT: fmul double
  // PRMTD-NEXT: fmul double
  // PRMTD-NEXT: fadd double
  // PRMTD-NEXT: fmul double
  // PRMTD-NEXT: fmul double
  // PRMTD-NEXT: fadd double
  // PRMTD-NEXT: fmul double
  // PRMTD-NEXT: fmul double
  // PRMTD-NEXT: fsub double
  // PRMTD-NEXT: fdiv double
  // PRMTD-NEXT: fdiv double
  // PRMTD-NEXT: fptrunc double {{.*}} to float
  // PRMTD-NEXT: fptrunc double {{.*}} to float

  return a / b;
}

// GH66555: float_control(precise) overrides the command-line complex range.
_Complex float pragma_precise_off_div(_Complex float a, _Complex float b) {
#pragma float_control(precise, off)
  // FULL-LABEL: @pragma_precise_off_div(
  // FULL: fdiv fast float
  // FULL-NEXT: fdiv fast float

  // BASIC-LABEL: @pragma_precise_off_div(
  // BASIC: fdiv fast float
  // BASIC-NEXT: fdiv fast float

  // IMPRVD-LABEL: @pragma_precise_off_div(
  // IMPRVD: fdiv fast float
  // IMPRVD-NEXT: fdiv fast float

  // PRMTD-LABEL: @pragma_precise_off_div(
  // PRMTD: fdiv fast float
  // PRMTD-NEXT: fdiv fast float

  // FAST-LABEL: @pragma_precise_off_div(
  // FAST: fdiv fast float
  // FAST-NEXT: fdiv fast float

  return a / b;
}

_Complex float pragma_precise_off_mul(_Complex float a, _Complex float b) {
#pragma float_control(precise, off)
  // FULL-LABEL: @pragma_precise_off_mul(
  // FULL: fmul fast float
  // FULL-NEXT: fmul fast float
  // FULL-NEXT: fmul fast float
  // FULL-NEXT: fmul fast float
  // FULL-NEXT: fsub fast float
  // FULL-NEXT: fadd fast float
  // FULL-NOT: @__mulsc3

  // BASIC-LABEL: @pragma_precise_off_mul(
  // BASIC: fmul fast float
  // BASIC-NEXT: fmul fast float
  // BASIC-NEXT: fmul fast float
  // BASIC-NEXT: fmul fast float
  // BASIC-NEXT: fsub fast float
  // BASIC-NEXT: fadd fast float
  // BASIC-NOT: @__mulsc3

  // IMPRVD-LABEL: @pragma_precise_off_mul(
  // IMPRVD: fmul fast float
  // IMPRVD-NEXT: fmul fast float
  // IMPRVD-NEXT: fmul fast float
  // IMPRVD-NEXT: fmul fast float
  // IMPRVD-NEXT: fsub fast float
  // IMPRVD-NEXT: fadd fast float
  // IMPRVD-NOT: @__mulsc3

  // PRMTD-LABEL: @pragma_precise_off_mul(
  // PRMTD: fmul fast float
  // PRMTD-NEXT: fmul fast float
  // PRMTD-NEXT: fmul fast float
  // PRMTD-NEXT: fmul fast float
  // PRMTD-NEXT: fsub fast float
  // PRMTD-NEXT: fadd fast float
  // PRMTD-NOT: @__mulsc3

  // FAST-LABEL: @pragma_precise_off_mul(
  // FAST: fmul fast float
  // FAST-NEXT: fmul fast float
  // FAST-NEXT: fmul fast float
  // FAST-NEXT: fmul fast float
  // FAST-NEXT: fsub fast float
  // FAST-NEXT: fadd fast float
  // FAST-NOT: @__mulsc3

  return a * b;
}

#pragma float_control(push)
#pragma float_control(precise, on)
_Complex float pragma_precise_on_div(_Complex float a, _Complex float b) {
  // FULL-LABEL: @pragma_precise_on_div(
  // FULL-NOT: fdiv
  // FULL: call {{.*}} @__divsc3
  // FULL-NOT: fdiv

  // BASIC-LABEL: @pragma_precise_on_div(
  // BASIC-NOT: fdiv
  // BASIC: call {{.*}} @__divsc3
  // BASIC-NOT: fdiv

  // IMPRVD-LABEL: @pragma_precise_on_div(
  // IMPRVD-NOT: fdiv
  // IMPRVD: call {{.*}} @__divsc3
  // IMPRVD-NOT: fdiv

  // PRMTD-LABEL: @pragma_precise_on_div(
  // PRMTD-NOT: fdiv
  // PRMTD: call {{.*}} @__divsc3
  // PRMTD-NOT: fdiv

  // FAST-LABEL: @pragma_precise_on_div(
  // FAST-NOT: fdiv
  // FAST: call {{.*}} @__divsc3
  // FAST-NOT: fdiv

  return a / b;
}

_Complex float pragma_precise_on_mul(_Complex float a, _Complex float b) {
  // FULL-LABEL: @pragma_precise_on_mul(
  // FULL: call {{.*}} @__mulsc3

  // BASIC-LABEL: @pragma_precise_on_mul(
  // BASIC: call {{.*}} @__mulsc3

  // IMPRVD-LABEL: @pragma_precise_on_mul(
  // IMPRVD: call {{.*}} @__mulsc3

  // PRMTD-LABEL: @pragma_precise_on_mul(
  // PRMTD: call {{.*}} @__mulsc3

  // FAST-LABEL: @pragma_precise_on_mul(
  // FAST: fmul float
  // FAST-NEXT: fmul float
  // FAST-NEXT: fmul float
  // FAST-NEXT: fmul float
  // FAST-NEXT: fsub float
  // FAST-NEXT: fadd float
  // FAST: call {{.*}} @__mulsc3

  return a * b;
}

void pragma_precise_on_div_assign(_Complex float *a, _Complex float b) {
  // FULL-LABEL: @pragma_precise_on_div_assign(
  // FULL-NOT: fdiv
  // FULL: call {{.*}} @__divsc3
  // FULL-NOT: fdiv

  // BASIC-LABEL: @pragma_precise_on_div_assign(
  // BASIC-NOT: fdiv
  // BASIC: call {{.*}} @__divsc3
  // BASIC-NOT: fdiv

  // IMPRVD-LABEL: @pragma_precise_on_div_assign(
  // IMPRVD-NOT: fdiv
  // IMPRVD: call {{.*}} @__divsc3
  // IMPRVD-NOT: fdiv

  // PRMTD-LABEL: @pragma_precise_on_div_assign(
  // PRMTD-NOT: fdiv
  // PRMTD: call {{.*}} @__divsc3
  // PRMTD-NOT: fdiv

  // FAST-LABEL: @pragma_precise_on_div_assign(
  // FAST-NOT: fdiv
  // FAST: call {{.*}} @__divsc3
  // FAST-NOT: fdiv

  *a /= b;
}
#pragma float_control(pop)
