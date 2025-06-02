// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-cir.ll
// RUN: FileCheck -input-file=%t-cir.ll %s

void test() {
  int i = 1;
  long l = 2l;
  float f = 3.0f;
  double d = 4.0;
  bool b1 = true;
  bool b2 = false;
  const int ci = 1;
  const long cl = 2l;
  const float cf = 3.0f;
  const double cd = 4.0;
  const bool cb1 = true;
  const bool cb2 = false;
  int uii;
  long uil;
  float uif;
  double uid;
  bool uib;
}

// CHECK: module
// CHECK: cir.func @_Z4testv()
// CHECK:    %[[I_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CHECK:    %[[L_PTR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["l", init] {alignment = 8 : i64}
// CHECK:    %[[F_PTR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f", init] {alignment = 4 : i64}
// CHECK:    %[[D_PTR:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["d", init] {alignment = 8 : i64}
// CHECK:    %[[B1_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b1", init] {alignment = 1 : i64}
// CHECK:    %[[B2_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b2", init] {alignment = 1 : i64}
// CHECK:    %[[CI_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ci", init, const] {alignment = 4 : i64}
// CHECK:    %[[CL_PTR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["cl", init, const] {alignment = 8 : i64}
// CHECK:    %[[CF_PTR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["cf", init, const] {alignment = 4 : i64}
// CHECK:    %[[CD_PTR:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["cd", init, const] {alignment = 8 : i64}
// CHECK:    %[[CB1_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cb1", init, const] {alignment = 1 : i64}
// CHECK:    %[[CB2_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cb2", init, const] {alignment = 1 : i64}
// CHECK:    %[[UII_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["uii"] {alignment = 4 : i64}
// CHECK:    %[[UIL_PTR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["uil"] {alignment = 8 : i64}
// CHECK:    %[[UIF_PTR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["uif"] {alignment = 4 : i64}
// CHECK:    %[[UID_PTR:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["uid"] {alignment = 8 : i64}
// CHECK:    %[[UIB_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["uib"] {alignment = 1 : i64}
// CHECK:    %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    cir.store align(4) %[[ONE]], %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[TWO:.*]] = cir.const #cir.int<2> : !s64i
// CHECK:    cir.store align(8) %[[TWO]], %[[L_PTR]] : !s64i, !cir.ptr<!s64i>
// CHECK:    %[[THREE:.*]] = cir.const #cir.fp<3.0{{.*}}> : !cir.float
// CHECK:    cir.store align(4) %[[THREE]], %[[F_PTR]] : !cir.float, !cir.ptr<!cir.float>
// CHECK:    %[[FOUR:.*]] = cir.const #cir.fp<4.0{{.*}}> : !cir.double
// CHECK:    cir.store align(8) %[[FOUR]], %[[D_PTR]] : !cir.double, !cir.ptr<!cir.double>
// CHECK:    %[[TRUE:.*]] = cir.const #true
// CHECK:    cir.store align(1) %[[TRUE]], %[[B1_PTR]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:    %[[FALSE:.*]] = cir.const #false
// CHECK:    cir.store align(1) %[[FALSE]], %[[B2_PTR]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:    %[[ONEC:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:    cir.store align(4) %[[ONEC]], %[[CI_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[TWOC:.*]] = cir.const #cir.int<2> : !s64i
// CHECK:    cir.store align(8) %[[TWOC]], %[[CL_PTR]] : !s64i, !cir.ptr<!s64i>
// CHECK:    %[[THREEC:.*]] = cir.const #cir.fp<3.0{{.*}}> : !cir.float
// CHECK:    cir.store align(4) %[[THREEC]], %[[CF_PTR]] : !cir.float, !cir.ptr<!cir.float>
// CHECK:    %[[FOURC:.*]] = cir.const #cir.fp<4.0{{.*}}> : !cir.double
// CHECK:    cir.store align(8) %[[FOURC]], %[[CD_PTR]] : !cir.double, !cir.ptr<!cir.double>
// CHECK:    %[[TRUEC:.*]] = cir.const #true
// CHECK:    cir.store align(1) %[[TRUEC]], %[[CB1_PTR]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:    %[[FALSEC:.*]] = cir.const #false
// CHECK:    cir.store align(1) %[[FALSEC]], %[[CB2_PTR]] : !cir.bool, !cir.ptr<!cir.bool>
