// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

#pragma acc routine seq
void GlobalFunc4();
#pragma acc routine(GlobalFunc4) seq

#pragma acc routine seq
#pragma acc routine seq
void GlobalFunc5();
#pragma acc routine(GlobalFunc5) seq
#pragma acc routine(GlobalFunc5) seq

void GlobalFunc6();
void GlobalFunc6();
#pragma acc routine(GlobalFunc6) seq
void GlobalFunc6(){}

void GlobalFunc7(){}
#pragma acc routine(GlobalFunc7) seq

void force_emit() {
  GlobalFunc4();
  GlobalFunc5();
  GlobalFunc6();
  GlobalFunc7();
}

// CHECK: cir.func{{.*}} @[[G6_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[G6_R_NAME:.*]]]>}
// CHECK: cir.func{{.*}} @[[G7_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[G7_R_NAME:.*]]]>}

// CHECK: cir.func{{.*}} @[[G4_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[G4_R_NAME:.*]], @[[G4_R2_NAME:.*]]]>}
// CHECK: cir.func{{.*}} @[[G5_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[G5_R_NAME:.*]], @[[G5_R1_NAME:.*]], @[[G5_R2_NAME:.*]], @[[G5_R3_NAME:.*]]]>}

// CHECK: acc.routine @[[G4_R_NAME]] func(@[[G4_NAME]]) seq
// CHECK: acc.routine @[[G5_R_NAME]] func(@[[G5_NAME]]) seq
// CHECK: acc.routine @[[G5_R1_NAME]] func(@[[G5_NAME]]) seq
//
// CHECK: acc.routine @[[G4_R2_NAME]] func(@[[G4_NAME]]) seq
//
// CHECK: acc.routine @[[G5_R2_NAME]] func(@[[G5_NAME]]) seq
// CHECK: acc.routine @[[G5_R3_NAME]] func(@[[G5_NAME]]) seq
//
// CHECK: acc.routine @[[G6_R_NAME]] func(@[[G6_NAME]]) seq
// CHECK: acc.routine @[[G7_R_NAME]] func(@[[G7_NAME]]) seq
