// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown -target-feature +avx512fp16 -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown -o - | FileCheck %s --check-prefix=X86

_Float16 _Complex add_half_rr(_Float16 a, _Float16 b) {
  // X86-LABEL: @add_half_rr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
_Float16 _Complex add_half_cr(_Float16 _Complex a, _Float16 b) {
  // X86-LABEL: @add_half_cr(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
_Float16 _Complex add_half_rc(_Float16 a, _Float16 _Complex b) {
  // X86-LABEL: @add_half_rc(
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}
_Float16 _Complex add_half_cc(_Float16 _Complex a, _Float16 _Complex b) {
  // X86-LABEL: @add_half_cc(
  // X86: fadd
  // X86: fadd
  // X86-NOT: fadd
  // X86: ret
  return a + b;
}

_Float16 _Complex sub_half_rr(_Float16 a, _Float16 b) {
  // X86-LABEL: @sub_half_rr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
_Float16 _Complex sub_half_cr(_Float16 _Complex a, _Float16 b) {
  // X86-LABEL: @sub_half_cr(
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
_Float16 _Complex sub_half_rc(_Float16 a, _Float16 _Complex b) {
  // X86-LABEL: @sub_half_rc(
  // X86: fsub
  // X86: fneg
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}
_Float16 _Complex sub_half_cc(_Float16 _Complex a, _Float16 _Complex b) {
  // X86-LABEL: @sub_half_cc(
  // X86: fsub
  // X86: fsub
  // X86-NOT: fsub
  // X86: ret
  return a - b;
}

_Float16 _Complex mul_half_rr(_Float16 a, _Float16 b) {
  // X86-LABEL: @mul_half_rr(
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
_Float16 _Complex mul_half_cr(_Float16 _Complex a, _Float16 b) {
  // X86-LABEL: @mul_half_cr(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
_Float16 _Complex mul_half_rc(_Float16 a, _Float16 _Complex b) {
  // X86-LABEL: @mul_half_rc(
  // X86: fmul
  // X86: fmul
  // X86-NOT: fmul
  // X86: ret
  return a * b;
}
_Float16 _Complex mul_half_cc(_Float16 _Complex a, _Float16 _Complex b) {
  // X86-LABEL: @mul_half_cc(
  // X86: %[[AC:[^ ]+]] = fmul
  // X86: %[[BD:[^ ]+]] = fmul
  // X86: %[[AD:[^ ]+]] = fmul
  // X86: %[[BC:[^ ]+]] = fmul
  // X86: %[[RR:[^ ]+]] = fsub half %[[AC]], %[[BD]]
  // X86: %[[RI:[^ ]+]] = fadd half
  // X86-DAG: %[[AD]]
  // X86-DAG: ,
  // X86-DAG: %[[BC]]
  // X86: fcmp uno half %[[RR]]
  // X86: fcmp uno half %[[RI]]
  // X86: call {{.*}} @__mulhc3(
  // X86: ret
  return a * b;
}

_Float16 _Complex div_half_rr(_Float16 a, _Float16 b) {
  // X86-LABEL: @div_half_rr(
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
_Float16 _Complex div_half_cr(_Float16 _Complex a, _Float16 b) {
  // X86-LABEL: @div_half_cr(
  // X86: fdiv
  // X86: fdiv
  // X86-NOT: fdiv
  // X86: ret
  return a / b;
}
_Float16 _Complex div_half_rc(_Float16 a, _Float16 _Complex b) {
  // X86-LABEL: @div_half_rc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divhc3(
  // X86: ret
  return a / b;
}
_Float16 _Complex div_half_cc(_Float16 _Complex a, _Float16 _Complex b) {
  // X86-LABEL: @div_half_cc(
  // X86-NOT: fdiv
  // X86: call {{.*}} @__divhc3(
  // X86: ret
  return a / b;
}
