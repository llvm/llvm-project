// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s --check-prefix=FULL

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=limited -o - | FileCheck --check-prefix=LMTD %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=smith -o - | FileCheck --check-prefix=SMITH %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -complex-range=extend -o - | FileCheck --check-prefix=EXTND %s

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

  // LMTD: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fsub float
  // LMTD-NEXT: fadd float

  // SMITH: fmul float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fsub float
  // SMITH-NEXT: fadd float

  // EXTND: fmul float
  // EXTND-NEXT: fmul float
  // EXTND-NEXT: fmul float
  // EXTND-NEXT: fmul float
  // EXTND-NEXT: fsub float
  // EXTND-NEXT: fadd float

  return a * b;
}

_Complex float pragma_off_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // LABEL: define {{.*}} @pragma_off_mul(

  // FULL: call {{.*}} @__mulsc3

  // LMTD: call {{.*}} @__mulsc3

  // SMITH: call {{.*}} @__mulsc3

  // EXTND: call {{.*}} @__mulsc3

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

  // SMITH: fmul float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fadd float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fadd float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fsub float
  // SMITH-NEXT: fdiv float
  // SMITH-NEXT: fdiv float

  // EXTND:   fpext float {{.*}} to double
  // EXTND:   fpext float {{.*}} to double
  // EXTND:   fmul double
  // EXTND:   fmul double
  // EXTND:   fadd double
  // EXTND:   fmul double
  // EXTND:   fmul double
  // EXTND:   fadd double
  // EXTND:   fmul double
  // EXTND:   fmul double
  // EXTND:   fsub double
  // EXTND:   fdiv double
  // EXTND:   fdiv double
  // EXTND:   fptrunc double
  // EXTND:   fptrunc double

  return a / b;
}

_Complex float pragma_off_div(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // LABEL: define {{.*}} @pragma_off_div(

  // FULL: call {{.*}} @__divsc3

  // LMTD: call {{.*}} @__divsc3

  // SMITH: call {{.*}} @__divsc3

  // EXTND: call {{.*}} @__divdc3

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

  // LMTD: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fmul float
  // LMTD-NEXT: fsub float
  // LMTD-NEXT: fadd float

  // SMITH: fmul float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fsub float
  // SMITH-NEXT: fadd float

  // EXTND: fmul float
  // EXTND-NEXT: fmul float
  // EXTND-NEXT: fmul float
  // EXTND-NEXT: fmul float
  // EXTND-NEXT: fsub float
  // EXTND-NEXT: fadd float

  return a * b;
}
_Complex float pragma_default_div(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE DEFAULT
  // LABEL: define {{.*}} @pragma_on_divx(

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

  // SMITH: call{{.*}}float @llvm.fabs.f32(float {{.*}})
  // SMITH-NEXT: call{{.*}}float @llvm.fabs.f32(float {{.*}})
  // SMITH-NEXT: fcmp{{.*}}ugt float {{.*}}, {{.*}}
  // SMITH-NEXT:   br i1 {{.*}}, label
  // SMITH:  abs_rhsr_greater_or_equal_abs_rhsi:
  // SMITH-NEXT: fdiv float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fadd float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fadd float
  // SMITH-NEXT: fdiv float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fsub float
  // SMITH-NEXT: fdiv float
  // SMITH-NEXT: br label
  // SMITH: abs_rhsr_less_than_abs_rhsi:
  // SMITH-NEXT: fdiv float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fadd float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fadd float
  // SMITH-NEXT: fdiv float
  // SMITH-NEXT: fmul float
  // SMITH-NEXT: fsub float
  // SMITH-NEXT: fdiv float

  // EXTND: load float, ptr {{.*}}
  // EXTND: fpext float {{.*}} to double
  // EXTND-NEXT: fpext float {{.*}} to double
  // EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 0
  // EXTND-NEXT: load float, ptr {{.*}}
  // EXTND-NEXT: getelementptr inbounds { float, float }, ptr {{.*}}, i32 0, i32 1
  // EXTND-NEXT: load float, ptr {{.*}}
  // EXTND-NEXT: fpext float {{.*}} to double
  // EXTND-NEXT: fpext float {{.*}} to double
  // EXTND-NEXT: fmul double
  // EXTND-NEXT: fmul double
  // EXTND-NEXT: fadd double
  // EXTND-NEXT: fmul double
  // EXTND-NEXT: fmul double
  // EXTND-NEXT: fadd double
  // EXTND-NEXT: fmul double
  // EXTND-NEXT: fmul double
  // EXTND-NEXT: fsub double
  // EXTND-NEXT: fdiv double
  // EXTND-NEXT: fdiv double
  // EXTND-NEXT: fptrunc double {{.*}} to float
  // EXTND-NEXT: fptrunc double {{.*}} to float

  return a / b;
}
