// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

#pragma acc routine nohost device_type(nvidia, radeon) seq
void Func1() {}
void Func2() {}
#pragma acc routine(Func2) device_type(radeon) seq

#pragma acc routine device_type(multicore) worker device_type(nvidia, radeon) seq
void Func3() {}
void Func4() {}
#pragma acc routine(Func4) device_type(nvidia) seq device_type(radeon) vector

#pragma acc routine device_type(multicore) gang device_type(nvidia, radeon) gang
void Func5() {}
void Func6() {}
#pragma acc routine(Func6) device_type(multicore) gang(dim:1) device_type(radeon) gang

#pragma acc routine device_type(host) gang device_type(nvidia, radeon) gang(dim:1)
void Func7() {}
void Func8() {}
#pragma acc routine(Func8) device_type(radeon) gang(dim:2)

#pragma acc routine device_type(nvidia) gang(dim:2) device_type(radeon) gang(dim:3)
void Func9() {}
void Func10() {}
#pragma acc routine(Func10) device_type(nvidia) gang device_type(radeon) gang(dim:3)

#pragma acc routine device_type(nvidia) gang(dim:2) device_type(radeon) gang(dim:3) device_type(multicore) gang
void Func11() {}
void Func12() {}
#pragma acc routine(Func12) device_type(nvidia) gang(dim:2) device_type(radeon) gang(dim:3)

#pragma acc routine device_type(nvidia) gang(dim:2) device_type(radeon) gang
void Func13() {}
void Func14() {}
#pragma acc routine(Func14) device_type(nvidia) gang(dim:2) device_type(radeon) gang

// CHECK: cir.func{{.*}} @[[F1_NAME:.*Func1[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F1_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F1_R_NAME]] func(@[[F1_NAME]]) seq ([#acc.device_type<nvidia>, #acc.device_type<radeon>]) nohost

// CHECK: cir.func{{.*}} @[[F2_NAME:.*Func2[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F2_R_NAME:.*]]]>}

// CHECK: cir.func{{.*}} @[[F3_NAME:.*Func3[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F3_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F3_R_NAME]] func(@[[F3_NAME]]) worker ([#acc.device_type<multicore>]) seq ([#acc.device_type<nvidia>, #acc.device_type<radeon>])

// CHECK: cir.func{{.*}} @[[F4_NAME:.*Func4[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F4_R_NAME:.*]]]>}

// CHECK: cir.func{{.*}} @[[F5_NAME:.*Func5[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F5_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F5_R_NAME]] func(@[[F5_NAME]]) gang([#acc.device_type<multicore>, #acc.device_type<nvidia>, #acc.device_type<radeon>])

// CHECK: cir.func{{.*}} @[[F6_NAME:.*Func6[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F6_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @[[F7_NAME:.*Func7[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F7_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F7_R_NAME]] func(@[[F7_NAME]]) gang([#acc.device_type<host>], dim: 1 : i64 [#acc.device_type<nvidia>], dim: 1 : i64 [#acc.device_type<radeon>])

// CHECK: cir.func{{.*}} @[[F8_NAME:.*Func8[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F8_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @[[F9_NAME:.*Func9[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F9_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F9_R_NAME]] func(@[[F9_NAME]]) gang(dim: 2 : i64 [#acc.device_type<nvidia>], dim: 3 : i64 [#acc.device_type<radeon>])
//
// CHECK: cir.func{{.*}} @[[F10_NAME:.*Func10[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F10_R_NAME:.*]]]>}

// CHECK: cir.func{{.*}} @[[F11_NAME:.*Func11[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F11_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F11_R_NAME]] func(@[[F11_NAME]]) gang([#acc.device_type<multicore>], dim: 2 : i64 [#acc.device_type<nvidia>], dim: 3 : i64 [#acc.device_type<radeon>])
//
// CHECK: cir.func{{.*}} @[[F12_NAME:.*Func12[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F12_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @[[F13_NAME:.*Func13[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F13_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F13_R_NAME]] func(@[[F13_NAME]]) gang([#acc.device_type<radeon>], dim: 2 : i64 [#acc.device_type<nvidia>])
//
// CHECK: cir.func{{.*}} @[[F14_NAME:.*Func14[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F14_R_NAME:.*]]]>}

// CHECK: acc.routine @[[F2_R_NAME]] func(@[[F2_NAME]]) seq ([#acc.device_type<radeon>]) 
// CHECK: acc.routine @[[F4_R_NAME]] func(@[[F4_NAME]]) vector ([#acc.device_type<radeon>]) seq ([#acc.device_type<nvidia>])
// CHECK: acc.routine @[[F6_R_NAME]] func(@[[F6_NAME]]) gang([#acc.device_type<radeon>], dim: 1 : i64 [#acc.device_type<multicore>])
// CHECK: acc.routine @[[F8_R_NAME]] func(@[[F8_NAME]]) gang(dim: 2 : i64 [#acc.device_type<radeon>])
// CHECK: acc.routine @[[F10_R_NAME]] func(@[[F10_NAME]]) gang([#acc.device_type<nvidia>], dim: 3 : i64 [#acc.device_type<radeon>])
// CHECK: acc.routine @[[F12_R_NAME]] func(@[[F12_NAME]]) gang(dim: 2 : i64 [#acc.device_type<nvidia>], dim: 3 : i64 [#acc.device_type<radeon>])
// CHECK: acc.routine @[[F14_R_NAME]] func(@[[F14_NAME]]) gang([#acc.device_type<radeon>], dim: 2 : i64 [#acc.device_type<nvidia>])
