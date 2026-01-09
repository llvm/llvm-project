// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

#pragma acc routine seq
auto Lambda1 = [](){};

auto Lambda2 = [](){};
#pragma acc routine(Lambda2) seq
#pragma acc routine(Lambda2) seq

#pragma acc routine seq
int GlobalFunc1();

int GlobalFunc2();
#pragma acc routine(GlobalFunc2) seq
#pragma acc routine(GlobalFunc1) seq

void force_emit() {
  Lambda1();
  Lambda2();
  GlobalFunc1();
  GlobalFunc2();
}

// CHECK: cir.func {{.*}}lambda{{.*}} @[[L1_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[L1_R_NAME:.*]]]>}
// CHECK: cir.func {{.*}}lambda{{.*}} @[[L2_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[L2_R_NAME:.*]], @[[L2_R2_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @[[G1_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[G1_R_NAME:.*]], @[[G1_R2_NAME:.*]]]>}
// CHECK: cir.func{{.*}} @[[G2_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[G2_R_NAME:.*]]]>}

// CHECK: acc.routine @[[L1_R_NAME]] func(@[[L1_NAME]]) seq
// CHECK: acc.routine @[[G1_R_NAME]] func(@[[G1_NAME]]) seq
// CHECK: acc.routine @[[L2_R_NAME]] func(@[[L2_NAME]]) seq
// CHECK: acc.routine @[[L2_R2_NAME]] func(@[[L2_NAME]]) seq
// CHECK: acc.routine @[[G2_R_NAME]] func(@[[G2_NAME]]) seq
// CHECK: acc.routine @[[G1_R2_NAME]] func(@[[G1_NAME]]) seq
