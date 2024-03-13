// RUN: %clang_cc1 %s -O0 -emit-llvm -triple x86_64-unknown-unknown -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -O0 -triple x86_64-unknown-unknown -fsyntax-only -ast-dump | FileCheck %s --check-prefix=AST

// Check that for 'F _Complex + int' (F = real floating-point type), we emit an
// implicit cast from 'int' to 'F', but NOT to 'F _Complex' (i.e. that we do
// 'F _Complex + F', NOT 'F _Complex + F _Complex'), and likewise for -/*.

// AST-NOT: FloatingRealToComplex

float _Complex add_float_ci(float _Complex a, int b) {
  // X86-LABEL: @add_float_ci
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to float
  // X86: fadd float {{.*}}, [[I]]
  // X86-NOT: fadd
  return a + b;
}

float _Complex add_float_ic(int a, float _Complex b) {
  // X86-LABEL: @add_float_ic
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to float
  // X86: fadd float [[I]]
  // X86-NOT: fadd
  return a + b;
}

float _Complex sub_float_ci(float _Complex a, int b) {
  // X86-LABEL: @sub_float_ci
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to float
  // X86: fsub float {{.*}}, [[I]]
  // X86-NOT: fsub
  return a - b;
}

float _Complex sub_float_ic(int a, float _Complex b) {
  // X86-LABEL: @sub_float_ic
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to float
  // X86: fsub float [[I]]
  // X86: fneg
  // X86-NOT: fsub
  return a - b;
}

float _Complex mul_float_ci(float _Complex a, int b) {
  // X86-LABEL: @mul_float_ci
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to float
  // X86: fmul float {{.*}}, [[I]]
  // X86: fmul float {{.*}}, [[I]]
  // X86-NOT: fmul
  return a * b;
}

float _Complex mul_float_ic(int a, float _Complex b) {
  // X86-LABEL: @mul_float_ic
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to float
  // X86: fmul float [[I]]
  // X86: fmul float [[I]]
  // X86-NOT: fmul
  return a * b;
}

float _Complex div_float_ci(float _Complex a, int b) {
  // X86-LABEL: @div_float_ci
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to float
  // X86: fdiv float {{.*}}, [[I]]
  // X86: fdiv float {{.*}}, [[I]]
  // X86-NOT: @__divsc3
  return a / b;
}

// There is no good way of doing this w/o converting the 'int' to a complex
// number, so we expect complex division here.
float _Complex div_float_ic(int a, float _Complex b) {
  // X86-LABEL: @div_float_ic
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to float
  // X86: call {{.*}} @__divsc3(float {{.*}} [[I]], float noundef 0.{{0+}}e+00, float {{.*}}, float {{.*}})
  return a / b;
}

double _Complex add_double_ci(double _Complex a, int b) {
  // X86-LABEL: @add_double_ci
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to double
  // X86: fadd double {{.*}}, [[I]]
  // X86-NOT: fadd
  return a + b;
}

double _Complex add_double_ic(int a, double _Complex b) {
  // X86-LABEL: @add_double_ic
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to double
  // X86: fadd double [[I]]
  // X86-NOT: fadd
  return a + b;
}

double _Complex sub_double_ci(double _Complex a, int b) {
  // X86-LABEL: @sub_double_ci
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to double
  // X86: fsub double {{.*}}, [[I]]
  // X86-NOT: fsub
  return a - b;
}

double _Complex sub_double_ic(int a, double _Complex b) {
  // X86-LABEL: @sub_double_ic
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to double
  // X86: fsub double [[I]]
  // X86: fneg
  // X86-NOT: fsub
  return a - b;
}

double _Complex mul_double_ci(double _Complex a, int b) {
  // X86-LABEL: @mul_double_ci
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to double
  // X86: fmul double {{.*}}, [[I]]
  // X86: fmul double {{.*}}, [[I]]
  // X86-NOT: fmul
  return a * b;
}

double _Complex mul_double_ic(int a, double _Complex b) {
  // X86-LABEL: @mul_double_ic
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to double
  // X86: fmul double [[I]]
  // X86: fmul double [[I]]
  // X86-NOT: fmul
  return a * b;
}

double _Complex div_double_ci(double _Complex a, int b) {
  // X86-LABEL: @div_double_ci
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to double
  // X86: fdiv double {{.*}}, [[I]]
  // X86: fdiv double {{.*}}, [[I]]
  // X86-NOT: @__divdc3
  return a / b;
}

// There is no good way of doing this w/o converting the 'int' to a complex
// number, so we expect complex division here.
double _Complex div_double_ic(int a, double _Complex b) {
  // X86-LABEL: @div_double_ic
  // X86: [[I:%.*]] = sitofp i32 {{%.*}} to double
  // X86: call {{.*}} @__divdc3(double {{.*}} [[I]], double noundef 0.{{0+}}e+00, double {{.*}}, double {{.*}})
  return a / b;
}
