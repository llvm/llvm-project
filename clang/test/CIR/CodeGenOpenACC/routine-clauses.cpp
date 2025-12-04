// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

#pragma acc routine seq nohost
void Func1() {}

void Func2() {}
#pragma acc routine(Func2) seq

#pragma acc routine worker
void Func3() {}

void Func4() {}
#pragma acc routine(Func4) worker nohost

#pragma acc routine nohost vector
void Func5() {}

void Func6() {}
#pragma acc routine(Func6) nohost vector

#pragma acc routine gang
void Func7() {}

void Func8() {}
#pragma acc routine(Func8) gang

#pragma acc routine gang(dim:1)
void Func9() {}

void Func10() {}
#pragma acc routine(Func10) gang(dim:3)

constexpr int Value = 2;

#pragma acc routine gang(dim:Value) nohost
void Func11() {}


void Func12() {}
#pragma acc routine(Func12) nohost gang(dim:Value)

// CHECK: cir.func{{.*}} @[[F1_NAME:.*Func1[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F1_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F1_R_NAME]] func(@[[F1_NAME]]) seq nohost

// CHECK: cir.func{{.*}} @[[F2_NAME:.*Func2[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F2_R_NAME:.*]]]>}

// CHECK: cir.func{{.*}} @[[F3_NAME:.*Func3[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F3_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F3_R_NAME]] func(@[[F3_NAME]]) worker

// CHECK: cir.func{{.*}} @[[F4_NAME:.*Func4[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F4_R_NAME:.*]]]>}

// CHECK: cir.func{{.*}} @[[F5_NAME:.*Func5[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F5_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F5_R_NAME]] func(@[[F5_NAME]]) vector

// CHECK: cir.func{{.*}} @[[F6_NAME:.*Func6[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F6_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @[[F7_NAME:.*Func7[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F7_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F7_R_NAME]] func(@[[F7_NAME]]) gang
//
// CHECK: cir.func{{.*}} @[[F8_NAME:.*Func8[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F8_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @[[F9_NAME:.*Func9[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F9_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F9_R_NAME]] func(@[[F9_NAME]]) gang(dim: 1 : i64)
//
// CHECK: cir.func{{.*}} @[[F10_NAME:.*Func10[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F10_R_NAME:.*]]]>}

// CHECK: cir.func{{.*}} @[[F11_NAME:.*Func11[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F11_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F11_R_NAME]] func(@[[F11_NAME]]) gang(dim: 2 : i64)
//
// CHECK: cir.func{{.*}} @[[F12_NAME:.*Func12[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F12_R_NAME:.*]]]>}

// CHECK: acc.routine @[[F2_R_NAME]] func(@[[F2_NAME]]) seq
// CHECK: acc.routine @[[F4_R_NAME]] func(@[[F4_NAME]]) worker nohost
// CHECK: acc.routine @[[F6_R_NAME]] func(@[[F6_NAME]]) vector nohost
// CHECK: acc.routine @[[F8_R_NAME]] func(@[[F8_NAME]]) gang
// CHECK: acc.routine @[[F10_R_NAME]] func(@[[F10_NAME]]) gang(dim: 3 : i64)
// CHECK: acc.routine @[[F12_R_NAME]] func(@[[F12_NAME]]) gang(dim: 2 : i64)
