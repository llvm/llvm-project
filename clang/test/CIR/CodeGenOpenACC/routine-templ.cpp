// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

#pragma acc routine seq
template<typename T>
void func(){}

void use() {
  func<int>();
  func<float>();
}

// CHECK: cir.func{{.*}} @[[T1_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[T1_R_NAME:.*]]]>}
// CHECK: cir.func{{.*}} @[[T2_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[T2_R_NAME:.*]]]>}
//
// CHECK: acc.routine @[[T1_R_NAME]] func(@[[T1_NAME]]) seq
// CHECK: acc.routine @[[T2_R_NAME]] func(@[[T2_NAME]]) seq
