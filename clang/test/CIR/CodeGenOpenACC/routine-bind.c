// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

#pragma acc routine seq bind("BIND1")
void Func1(){}

void Func2(){}
#pragma acc routine(Func2) seq bind("BIND2")

#pragma acc routine seq device_type(nvidia) bind("BIND3")
void Func3(){}

void Func4(){}
#pragma acc routine(Func4) seq device_type(radeon) bind("BIND4")

#pragma acc routine seq device_type(nvidia, host) bind("BIND5_N") device_type(multicore) bind("BIND5_M")
void Func5(){}

void Func6(){}
#pragma acc routine(Func6) seq device_type(radeon) bind("BIND6_R") device_type(multicore, host) bind("BIND6_M")

// CHECK: cir.func{{.*}} @[[F1_NAME:.*Func1[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F1_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F1_R_NAME]] func(@[[F1_NAME]]) bind("BIND1") seq
//
// CHECK: cir.func{{.*}} @[[F2_NAME:.*Func2[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F2_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @[[F3_NAME:.*Func3[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F3_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F3_R_NAME]] func(@[[F3_NAME]]) bind("BIND3" [#acc.device_type<nvidia>]) seq
//
// CHECK: cir.func{{.*}} @[[F4_NAME:.*Func4[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F4_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @[[F5_NAME:.*Func5[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F5_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F5_R_NAME]] func(@[[F5_NAME]]) bind("BIND5_N" [#acc.device_type<nvidia>], "BIND5_N" [#acc.device_type<host>], "BIND5_M" [#acc.device_type<multicore>]) seq
//
// CHECK: cir.func{{.*}} @[[F6_NAME:.*Func6[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F6_R_NAME:.*]]]>}
//
// CHECK: acc.routine @[[F2_R_NAME]] func(@[[F2_NAME]]) bind("BIND2") seq
// CHECK: acc.routine @[[F4_R_NAME]] func(@[[F4_NAME]]) bind("BIND4" [#acc.device_type<radeon>]) seq
// CHECK: acc.routine @[[F6_R_NAME]] func(@[[F6_NAME]]) bind("BIND6_R" [#acc.device_type<radeon>], "BIND6_M" [#acc.device_type<multicore>], "BIND6_M" [#acc.device_type<host>]) seq

