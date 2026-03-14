// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

struct S {
#pragma acc routine seq
  void MemFunc1();
  void MemFunc2();
#pragma acc routine(S::MemFunc2) seq
  void MemFunc3();
#pragma acc routine(S::MemFunc3) seq

#pragma acc routine seq
  static void StaticMemFunc1();
  static void StaticMemFunc2();
  static void StaticMemFunc3();
#pragma acc routine(StaticMemFunc3) seq

#pragma acc routine seq
  static constexpr auto StaticLambda1 = [](){};
 static constexpr auto StaticLambda2 = [](){};
};
#pragma acc routine(S::MemFunc2) seq
#pragma acc routine(S::StaticLambda2) seq
#pragma acc routine(S::StaticMemFunc2) seq

void force_emit() {
  S{}.MemFunc1();
  S{}.MemFunc2();
  S{}.MemFunc3();
  S::StaticMemFunc1();
  S::StaticMemFunc2();
  S::StaticMemFunc3();
  S::StaticLambda1();
  S::StaticLambda2();
}

// CHECK: cir.func{{.*}} @[[MEM1_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[MEM1_R_NAME:.*]]]>}
// CHECK: cir.func{{.*}} @[[MEM2_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[MEM2_R_NAME:.*]], @[[MEM2_R2_NAME:.*]]]>}
// CHECK: cir.func{{.*}} @[[MEM3_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[MEM3_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @[[STATICMEM1_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[STATICMEM1_R_NAME:.*]]]>}
// CHECK: cir.func{{.*}} @[[STATICMEM2_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[STATICMEM2_R_NAME:.*]]]>}
// CHECK: cir.func{{.*}} @[[STATICMEM3_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[STATICMEM3_R_NAME:.*]]]>}
//
// CHECK: cir.func {{.*}}lambda{{.*}} @[[L1_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[L1_R_NAME:.*]]]>}
// CHECK: cir.func {{.*}}lambda{{.*}} @[[L2_NAME:[^\(]*]]({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[L2_R_NAME:.*]]]>}
//
// CHECK: acc.routine @[[MEM1_R_NAME]] func(@[[MEM1_NAME]]) seq
// CHECK: acc.routine @[[STATICMEM1_R_NAME]] func(@[[STATICMEM1_NAME]]) seq
// CHECK: acc.routine @[[L1_R_NAME]] func(@[[L1_NAME]]) seq
// CHECK: acc.routine @[[MEM2_R_NAME]] func(@[[MEM2_NAME]]) seq
// CHECK: acc.routine @[[MEM3_R_NAME]] func(@[[MEM3_NAME]]) seq
// CHECK: acc.routine @[[STATICMEM3_R_NAME]] func(@[[STATICMEM3_NAME]]) seq
// CHECK: acc.routine @[[MEM2_R2_NAME]] func(@[[MEM2_NAME]]) seq
// CHECK: acc.routine @[[L2_R_NAME]] func(@[[L2_NAME]]) seq
// CHECK: acc.routine @[[STATICMEM2_R_NAME]] func(@[[STATICMEM2_NAME]]) seq
