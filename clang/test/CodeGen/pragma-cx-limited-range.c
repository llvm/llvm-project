// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -o - | FileCheck %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fcx-limited-range -o - | FileCheck --check-prefix=OPT-CXLMT %s

// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown \
// RUN: -fno-cx-limited-range -o - | FileCheck %s

_Complex float pragma_on_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE ON
  // CHECK-LABEL: define {{.*}} @pragma_on_mul(
  // CHECK: fmul float
  // CHECK-NEXT: fmul float
  // CHECK-NEXT: fsub float
  // CHECK-NEXT: fmul float
  // CHECK-NEXT: fmul float
  // CHECK-NEXT: fadd float

  // OPT-CXLMT-LABEL: define {{.*}} @pragma_on_mul(
  // OPT-CXLMT: fmul float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fsub float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fadd float
  return a * b;
}

_Complex float pragma_off_mul(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // CHECK-LABEL: define {{.*}} @pragma_off_mul(
  // CHECK: call {{.*}} @__mulsc3

  // OPT-CXLMT-LABEL: define {{.*}} @pragma_off_mul(
  // OPT-CXLMT: fmul float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fsub float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fadd float
  return a * b;
}

_Complex float pragma_on_div(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE ON
  // CHECK-LABEL: define {{.*}} @pragma_on_div(
  // CHECK: fmul float
  // CHECK-NEXT: fmul float
  // CHECK-NEXT: fadd float
  // CHECK-NEXT: fmul float
  // CHECK-NEXT: fmul float
  // CHECK-NEXT: fadd float
  // CHECK-NEXT: fmul float
  // CHECK-NEXT: fmul float
  // CHECK-NEXT: fsub float
  // CHECK-NEXT: fdiv float
  // CHECK: fdiv float

  // OPT-CXLMT-LABEL: define {{.*}} @pragma_on_div(
  // OPT-CXLMT: fmul float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fadd float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fadd float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fsub float
  // OPT-CXLMT-NEXT: fdiv float
  // OPT-CXLMT-NEXT: fdiv float
  return a / b;
}

_Complex float pragma_off_div(_Complex float a, _Complex float b) {
#pragma STDC CX_LIMITED_RANGE OFF
  // CHECK-LABEL: define {{.*}} @pragma_off_div(
  // CHECK: call {{.*}} @__divsc3

  // OPT-CXLMT-LABEL: define {{.*}} @pragma_off_div(
  // OPT-CXLMT: fmul float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fadd float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fadd float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fmul float
  // OPT-CXLMT-NEXT: fsub float
  // OPT-CXLMT-NEXT: fdiv float
  // OPT-CXLMT-NEXT: fdiv float
  return a / b;
}
