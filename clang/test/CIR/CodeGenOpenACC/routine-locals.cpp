// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

void GlobalFunc();
void InFunc() {

#pragma acc routine(GlobalFunc) seq
  GlobalFunc();

#pragma acc routine seq
  auto Lambda1 = [](){};
  Lambda1();

  auto Lambda2 = [](){};
#pragma acc routine(Lambda2) seq
  Lambda2();
};

// CHECK: cir.func{{.*}} @[[G1_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[G1_R_NAME:.*]]]>}
// CHECK: cir.func {{.*}}lambda{{.*}} @[[L1_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[L1_R_NAME:.*]]]>}
// CHECK: cir.func {{.*}}lambda{{.*}} @[[L2_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[L2_R_NAME:.*]]]>}

// CHECK: acc.routine @[[L1_R_NAME]] func(@[[L1_NAME]]) seq
// CHECK: acc.routine @[[G1_R_NAME]] func(@[[G1_NAME]]) seq
// CHECK: acc.routine @[[L2_R_NAME]] func(@[[L2_NAME]]) seq
