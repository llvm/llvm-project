// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

namespace NS1 {
#pragma acc routine seq
  int NSFunc1();
#pragma acc routine seq
  auto Lambda1 = [](){};

  auto Lambda2 = [](){};
} // namespace NS1

#pragma acc routine(NS1::NSFunc1) seq
#pragma acc routine(NS1::Lambda2) seq

void force_emit() {
  NS1::NSFunc1();
  NS1::Lambda1();
  NS1::Lambda2();
}

// CHECK: cir.func{{.*}} @[[F1_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F1_R_NAME:.*]], @[[F1_R2_NAME:.*]]]>}
// CHECK: cir.func {{.*}}lambda{{.*}} @[[L1_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[L1_R_NAME:.*]]]>}
// CHECK: cir.func {{.*}}lambda{{.*}} @[[L2_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[L2_R_NAME:.*]]]>}
//
// CHECK: acc.routine @[[F1_R_NAME]] func(@[[F1_NAME]]) seq
// CHECK: acc.routine @[[L1_R_NAME]] func(@[[L1_NAME]]) seq
// CHECK: acc.routine @[[F1_R2_NAME]] func(@[[F1_NAME]]) seq
// CHECK: acc.routine @[[L2_R_NAME]] func(@[[L2_NAME]]) seq
