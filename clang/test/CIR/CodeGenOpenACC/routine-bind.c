// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s
// FIXME: We should run this against Windows mangling as well at one point.

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

#pragma acc routine seq bind(BIND7)
void Func7(int i){}

void Func8(float f){}
#pragma acc routine(Func8) seq bind(BIND8)

#pragma acc routine seq device_type(nvidia) bind(BIND9)
void Func9(int i, float f, short s){}

struct S{};
struct U{};
struct V{};

void Func10(struct S s){}
#pragma acc routine(Func10) seq device_type(radeon) bind(BIND10)

#pragma acc routine seq device_type(nvidia, host) bind(BIND11_NVH) device_type(multicore) bind(BIND11_MC)
void Func11(struct U* u, struct V v, int i){}

int Func12(struct U u, struct V v, int i){ return 0; }
#pragma acc routine(Func12) seq device_type(radeon) bind(BIND12_R) device_type(multicore, host) bind(BIND12_MCH)

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
// CHECK: cir.func{{.*}} @[[F7_NAME:.*Func7[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F7_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F7_R_NAME]] func(@[[F7_NAME]]) bind(@BIND7) seq
//
// CHECK: cir.func{{.*}} @[[F8_NAME:.*Func8[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F8_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @[[F9_NAME:.*Func9[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F9_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F9_R_NAME]] func(@[[F9_NAME]]) bind(@BIND9 [#acc.device_type<nvidia>]) seq
//
// CHECK: cir.func{{.*}} @[[F10_NAME:.*Func10[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F10_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @[[F11_NAME:.*Func11[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F11_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F11_R_NAME]] func(@[[F11_NAME]]) bind(@BIND11_NVH [#acc.device_type<nvidia>], @BIND11_NVH [#acc.device_type<host>], @BIND11_MC [#acc.device_type<multicore>])
//
// CHECK: cir.func{{.*}} @[[F12_NAME:.*Func12[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F12_R_NAME:.*]]]>}
//
// CHECK: acc.routine @[[F2_R_NAME]] func(@[[F2_NAME]]) bind("BIND2") seq
// CHECK: acc.routine @[[F4_R_NAME]] func(@[[F4_NAME]]) bind("BIND4" [#acc.device_type<radeon>]) seq
// CHECK: acc.routine @[[F6_R_NAME]] func(@[[F6_NAME]]) bind("BIND6_R" [#acc.device_type<radeon>], "BIND6_M" [#acc.device_type<multicore>], "BIND6_M" [#acc.device_type<host>]) seq

// CHECK: acc.routine @[[F8_R_NAME]] func(@[[F8_NAME]]) bind(@BIND8) seq
// CHECK: acc.routine @[[F10_R_NAME]] func(@[[F10_NAME]]) bind(@BIND10 [#acc.device_type<radeon>]) seq
// CHECK: acc.routine @[[F12_R_NAME]] func(@[[F12_NAME]]) bind(@BIND12_R [#acc.device_type<radeon>], @BIND12_MCH [#acc.device_type<multicore>], @BIND12_MCH [#acc.device_type<host>]) seq
