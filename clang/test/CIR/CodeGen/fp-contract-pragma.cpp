// ClangIR port of clang/test/CodeGen/fp-contract-pragma.cpp.
// The CIR-lowered and classic CodeGen LLVM IR match here, so both feed LLVM.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -Wno-unused-value -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -Wno-unused-value -emit-llvm %s -o %t-ogcg.ll
// RUN: FileCheck --input-file=%t-ogcg.ll %s -check-prefix=LLVM

// Is FP_CONTRACT honored in a simple case?
float fp_contract_1(float a, float b, float c) {
  #pragma STDC FP_CONTRACT ON
  return a * b + c;
}
// CIR-LABEL: cir.func {{.*}}@_Z13fp_contract_1fff
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// CIR-NOT: cir.fmul
// LLVM-LABEL: @_Z13fp_contract_1fff
// LLVM: call float @llvm.fmuladd.f32

// Is FP_CONTRACT state cleared on exiting compound statements?
float fp_contract_2(float a, float b, float c) {
  {
  #pragma STDC FP_CONTRACT ON
  }
  return a * b + c;
}
// CIR-LABEL: cir.func {{.*}}@_Z13fp_contract_2fff
// CIR: cir.fmul %{{.*}}, %{{.*}} : !cir.float
// CIR: cir.fadd %{{.*}}, %{{.*}} : !cir.float
// CIR-NOT: cir.fmuladd
// LLVM-LABEL: @_Z13fp_contract_2fff
// LLVM: %[[M:.*]] = fmul float
// LLVM: fadd float %[[M]],

// Does FP_CONTRACT survive template instantiation?
class Foo {};
Foo operator+(Foo, Foo);

template <typename T>
T template_muladd(T a, T b, T c) {
  #pragma STDC FP_CONTRACT ON
  return a * b + c;
}
// The fmuladd is emitted in the instantiated template body.
// CIR-LABEL: cir.func {{.*}}@_Z15template_muladdIfET_S0_S0_S0_
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// LLVM-LABEL: @_Z15template_muladdIfET_S0_S0_S0_
// LLVM: call {{.*}}float @llvm.fmuladd.f32

// fp_contract_3 is just a caller; the fused op lives in the instantiated
// template_muladd checked above. It is emitted in a different order under the
// classic CodeGen path, so it carries no checks of its own here.
float fp_contract_3(float a, float b, float c) {
  return template_muladd<float>(a, b, c);
}

template<typename T> class fp_contract_4 {
  float method(float a, float b, float c) {
    #pragma STDC FP_CONTRACT ON
    return a * b + c;
  }
};
template class fp_contract_4<int>;
// CIR-LABEL: cir.func {{.*}}@_ZN13fp_contract_4IiE6methodEfff
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// LLVM-LABEL: @_ZN13fp_contract_4IiE6methodEfff
// LLVM: call float @llvm.fmuladd.f32

// Check file-scoped FP_CONTRACT
#pragma STDC FP_CONTRACT ON
float fp_contract_5(float a, float b, float c) {
  return a * b + c;
}
// CIR-LABEL: cir.func {{.*}}@_Z13fp_contract_5fff
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// LLVM-LABEL: @_Z13fp_contract_5fff
// LLVM: call float @llvm.fmuladd.f32

#pragma STDC FP_CONTRACT OFF
float fp_contract_6(float a, float b, float c) {
  return a * b + c;
}
// CIR-LABEL: cir.func {{.*}}@_Z13fp_contract_6fff
// CIR: cir.fmul %{{.*}}, %{{.*}} : !cir.float
// CIR: cir.fadd %{{.*}}, %{{.*}} : !cir.float
// CIR-NOT: cir.fmuladd
// LLVM-LABEL: @_Z13fp_contract_6fff
// LLVM: %[[M:.*]] = fmul float
// LLVM: fadd float %[[M]],

// If the multiply has multiple uses, don't produce fmuladd.
// This used to assert (PR25719):
// https://llvm.org/bugs/show_bug.cgi?id=25719
float fp_contract_7(float a, float b, float c) {
  #pragma STDC FP_CONTRACT ON
  return (a = 2 * b) - c;
}
// CIR-LABEL: cir.func {{.*}}@_Z13fp_contract_7fff
// CIR: cir.fmul %{{.*}}, %{{.*}} : !cir.float
// CIR: cir.fsub %{{.*}}, %{{.*}} : !cir.float
// CIR-NOT: cir.fmuladd
// LLVM-LABEL: @_Z13fp_contract_7fff
// LLVM: %[[M:.*]] = fmul float
// LLVM: fsub float %[[M]],

// a * b - c  =>  fmuladd(a, b, -c)
float fp_contract_8(float a, float b, float c) {
  #pragma STDC FP_CONTRACT ON
  return a * b - c;
}
// CIR-LABEL: cir.func {{.*}}@_Z13fp_contract_8fff
// CIR: cir.fneg %{{.*}} : !cir.float
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// LLVM-LABEL: @_Z13fp_contract_8fff
// LLVM: fneg float
// LLVM: call float @llvm.fmuladd.f32

// c - a * b  =>  fmuladd(-a, b, c)  (mul on the RHS of a subtraction)
float fp_contract_9(float a, float b, float c) {
  #pragma STDC FP_CONTRACT ON
  return c - a * b;
}
// CIR-LABEL: cir.func {{.*}}@_Z13fp_contract_9fff
// CIR: cir.fneg %{{.*}} : !cir.float
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// LLVM-LABEL: @_Z13fp_contract_9fff
// LLVM: fneg float
// LLVM: call float @llvm.fmuladd.f32

// -(a * b) + c  =>  fmuladd(-a, b, c)  (peek through fneg on the LHS)
float fp_contract_10(float a, float b, float c) {
  #pragma STDC FP_CONTRACT ON
  return -(a * b) + c;
}
// CIR-LABEL: cir.func {{.*}}@_Z14fp_contract_10fff
// CIR: cir.fneg %{{.*}} : !cir.float
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// CIR-NOT: cir.fadd
// LLVM-LABEL: @_Z14fp_contract_10fff
// LLVM: fneg float
// LLVM: call float @llvm.fmuladd.f32

// -(a * b) - c  =>  fmuladd(-a, b, -c)  (fneg both the mul operand and addend)
float fp_contract_11(float a, float b, float c) {
  #pragma STDC FP_CONTRACT ON
  return -(a * b) - c;
}
// CIR-LABEL: cir.func {{.*}}@_Z14fp_contract_11fff
// CIR: cir.fneg %{{.*}} : !cir.float
// CIR: cir.fneg %{{.*}} : !cir.float
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// LLVM-LABEL: @_Z14fp_contract_11fff
// LLVM: fneg float
// LLVM: fneg float
// LLVM: call float @llvm.fmuladd.f32

// c + -(a * b)  =>  fmuladd(-a, b, c)  (peek through fneg on the RHS)
float fp_contract_12(float a, float b, float c) {
  #pragma STDC FP_CONTRACT ON
  return c + -(a * b);
}
// CIR-LABEL: cir.func {{.*}}@_Z14fp_contract_12fff
// CIR: cir.fneg %{{.*}} : !cir.float
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// CIR-NOT: cir.fadd
// LLVM-LABEL: @_Z14fp_contract_12fff
// LLVM: fneg float
// LLVM: call float @llvm.fmuladd.f32

// c - -(a * b)  =>  fmuladd(a, b, c)  (the two negations cancel; no fneg)
float fp_contract_13(float a, float b, float c) {
  #pragma STDC FP_CONTRACT ON
  return c - -(a * b);
}
// CIR-LABEL: cir.func {{.*}}@_Z14fp_contract_13fff
// CIR-NOT: cir.fneg
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// LLVM-LABEL: @_Z14fp_contract_13fff
// LLVM-NOT: fneg float
// LLVM: call float @llvm.fmuladd.f32

// Mul reused by the assignment, so no fusion. At -O0 the negation stays an
// fneg+fadd instead of the fsub the -O3 original test expects.
float fp_contract_14(float a, float b, float c) {
  #pragma STDC FP_CONTRACT ON
  float d;
  return (d = -(a * b)) + c;
}
// CIR-LABEL: cir.func {{.*}}@_Z14fp_contract_14fff
// CIR: cir.fmul %{{.*}}, %{{.*}} : !cir.float
// CIR: cir.fneg %{{.*}} : !cir.float
// CIR: cir.fadd %{{.*}}, %{{.*}} : !cir.float
// CIR-NOT: cir.fmuladd
// LLVM-LABEL: @_Z14fp_contract_14fff
// LLVM: fmul float
// LLVM: fneg float
// LLVM: fadd float

// Same as above, with the negation applied to the assignment result.
float fp_contract_15(float a, float b, float c) {
  #pragma STDC FP_CONTRACT ON
  float d;
  return -(d = (a * b)) + c;
}
// CIR-LABEL: cir.func {{.*}}@_Z14fp_contract_15fff
// CIR: cir.fmul %{{.*}}, %{{.*}} : !cir.float
// CIR: cir.fneg %{{.*}} : !cir.float
// CIR: cir.fadd %{{.*}}, %{{.*}} : !cir.float
// CIR-NOT: cir.fmuladd
// LLVM-LABEL: @_Z14fp_contract_15fff
// LLVM: fmul float
// LLVM: fneg float
// LLVM: fadd float
