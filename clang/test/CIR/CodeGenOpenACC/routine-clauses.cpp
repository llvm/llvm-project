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

// CHECK: cir.func{{.*}} @[[F1_NAME:.*Func1[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F1_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F1_R_NAME]] func(@[[F1_NAME]]) seq nohost

// CHECK: cir.func{{.*}} @[[F2_NAME:.*Func2[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F2_R_NAME:.*]]]>}

// CHECK: cir.func{{.*}} @[[F3_NAME:.*Func3[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F3_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F3_R_NAME]] func(@[[F3_NAME]]) worker

// CHECK: cir.func{{.*}} @[[F4_NAME:.*Func4[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F4_R_NAME:.*]]]>}

// CHECK: cir.func{{.*}} @[[F5_NAME:.*Func5[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F5_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F5_R_NAME]] func(@[[F5_NAME]]) vector

// CHECK: cir.func{{.*}} @[[F6_NAME:.*Func6[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F6_R_NAME:.*]]]>}

// CHECK: acc.routine @[[F2_R_NAME]] func(@[[F2_NAME]]) seq
// CHECK: acc.routine @[[F4_R_NAME]] func(@[[F4_NAME]]) worker nohost
// CHECK: acc.routine @[[F6_R_NAME]] func(@[[F6_NAME]]) vector nohost
