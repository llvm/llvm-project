// ClangIR port of clang/test/CodeGen/fp-contract-on-pragma.cpp.
// The CIR-lowered and classic CodeGen LLVM IR match here, so both feed LLVM.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -Wno-unused-value -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -Wno-unused-value -emit-llvm %s -o %t-ogcg.ll
// RUN: FileCheck --input-file=%t-ogcg.ll %s -check-prefix=LLVM

// Is FP_CONTRACT honored in a simple case?
float fp_contract_1(float a, float b, float c) {
#pragma clang fp contract(on)
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
#pragma clang fp contract(on)
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
#pragma clang fp contract(on)
  return a * b + c;
}
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

template <typename T>
class fp_contract_4 {
  float method(float a, float b, float c) {
#pragma clang fp contract(on)
    return a * b + c;
  }
};
template class fp_contract_4<int>;
// CIR-LABEL: cir.func {{.*}}@_ZN13fp_contract_4IiE6methodEfff
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// LLVM-LABEL: @_ZN13fp_contract_4IiE6methodEfff
// LLVM: call float @llvm.fmuladd.f32

// Check file-scoped FP_CONTRACT
#pragma clang fp contract(on)
float fp_contract_5(float a, float b, float c) {
  return a * b + c;
}
// CIR-LABEL: cir.func {{.*}}@_Z13fp_contract_5fff
// CIR: cir.fmuladd %{{.*}}, %{{.*}}, %{{.*}} : !cir.float
// LLVM-LABEL: @_Z13fp_contract_5fff
// LLVM: call float @llvm.fmuladd.f32

#pragma clang fp contract(off)
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
#pragma clang fp contract(on)
  return (a = 2 * b) - c;
}
// CIR-LABEL: cir.func {{.*}}@_Z13fp_contract_7fff
// CIR: cir.fmul %{{.*}}, %{{.*}} : !cir.float
// CIR: cir.fsub %{{.*}}, %{{.*}} : !cir.float
// CIR-NOT: cir.fmuladd
// LLVM-LABEL: @_Z13fp_contract_7fff
// LLVM: %[[M:.*]] = fmul float
// LLVM: fsub float %[[M]],

// contract(on) only fuses within a statement: a mul and add in separate
// statements are not contracted.
float fp_contract_8(float a, float b, float c) {
#pragma clang fp contract(on)
  float t = a * b;
  return t + c;
}
// CIR-LABEL: cir.func {{.*}}@_Z13fp_contract_8fff
// CIR: cir.fmul %{{.*}}, %{{.*}} : !cir.float
// CIR: cir.fadd %{{.*}}, %{{.*}} : !cir.float
// CIR-NOT: cir.fmuladd
// LLVM-LABEL: @_Z13fp_contract_8fff
// LLVM: fmul float
// LLVM: fadd float
// LLVM-NOT: call float @llvm.fmuladd.f32
