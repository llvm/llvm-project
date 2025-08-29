// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

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

// CIR: module
// CIR: cir.func{{.*}} @_Z4testv()
// CIR:    %[[I_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR:    %[[L_PTR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["l", init] {alignment = 8 : i64}
// CIR:    %[[F_PTR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f", init] {alignment = 4 : i64}
// CIR:    %[[D_PTR:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["d", init] {alignment = 8 : i64}
// CIR:    %[[B1_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b1", init] {alignment = 1 : i64}
// CIR:    %[[B2_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b2", init] {alignment = 1 : i64}
// CIR:    %[[CI_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ci", init, const] {alignment = 4 : i64}
// CIR:    %[[CL_PTR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["cl", init, const] {alignment = 8 : i64}
// CIR:    %[[CF_PTR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["cf", init, const] {alignment = 4 : i64}
// CIR:    %[[CD_PTR:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["cd", init, const] {alignment = 8 : i64}
// CIR:    %[[CB1_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cb1", init, const] {alignment = 1 : i64}
// CIR:    %[[CB2_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cb2", init, const] {alignment = 1 : i64}
// CIR:    %[[UII_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["uii"] {alignment = 4 : i64}
// CIR:    %[[UIL_PTR:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["uil"] {alignment = 8 : i64}
// CIR:    %[[UIF_PTR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["uif"] {alignment = 4 : i64}
// CIR:    %[[UID_PTR:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["uid"] {alignment = 8 : i64}
// CIR:    %[[UIB_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["uib"] {alignment = 1 : i64}
// CIR:    %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:    cir.store align(4) %[[ONE]], %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR:    %[[TWO:.*]] = cir.const #cir.int<2> : !s64i
// CIR:    cir.store align(8) %[[TWO]], %[[L_PTR]] : !s64i, !cir.ptr<!s64i>
// CIR:    %[[THREE:.*]] = cir.const #cir.fp<3.0{{.*}}> : !cir.float
// CIR:    cir.store align(4) %[[THREE]], %[[F_PTR]] : !cir.float, !cir.ptr<!cir.float>
// CIR:    %[[FOUR:.*]] = cir.const #cir.fp<4.0{{.*}}> : !cir.double
// CIR:    cir.store align(8) %[[FOUR]], %[[D_PTR]] : !cir.double, !cir.ptr<!cir.double>
// CIR:    %[[TRUE:.*]] = cir.const #true
// CIR:    cir.store align(1) %[[TRUE]], %[[B1_PTR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:    %[[FALSE:.*]] = cir.const #false
// CIR:    cir.store align(1) %[[FALSE]], %[[B2_PTR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:    %[[ONEC:.*]] = cir.const #cir.int<1> : !s32i
// CIR:    cir.store align(4) %[[ONEC]], %[[CI_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR:    %[[TWOC:.*]] = cir.const #cir.int<2> : !s64i
// CIR:    cir.store align(8) %[[TWOC]], %[[CL_PTR]] : !s64i, !cir.ptr<!s64i>
// CIR:    %[[THREEC:.*]] = cir.const #cir.fp<3.0{{.*}}> : !cir.float
// CIR:    cir.store align(4) %[[THREEC]], %[[CF_PTR]] : !cir.float, !cir.ptr<!cir.float>
// CIR:    %[[FOURC:.*]] = cir.const #cir.fp<4.0{{.*}}> : !cir.double
// CIR:    cir.store align(8) %[[FOURC]], %[[CD_PTR]] : !cir.double, !cir.ptr<!cir.double>
// CIR:    %[[TRUEC:.*]] = cir.const #true
// CIR:    cir.store align(1) %[[TRUEC]], %[[CB1_PTR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:    %[[FALSEC:.*]] = cir.const #false
// CIR:    cir.store align(1) %[[FALSEC]], %[[CB2_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: define dso_local void @_Z4testv()
// LLVM:   %[[I_PTR:.+]] = alloca i32
// LLVM:   %[[L_PTR:.+]] = alloca i64
// LLVM:   %[[F_PTR:.+]] = alloca float
// LLVM:   %[[D_PTR:.+]] = alloca double
// LLVM:   %[[B1_PTR:.+]] = alloca i8
// LLVM:   %[[B2_PTR:.+]] = alloca i8
// LLVM:   %[[CI_PTR:.+]] = alloca i32
// LLVM:   %[[CL_PTR:.+]] = alloca i64
// LLVM:   %[[CF_PTR:.+]] = alloca float
// LLVM:   %[[CD_PTR:.+]] = alloca double
// LLVM:   %[[CB1_PTR:.+]] = alloca i8
// LLVM:   %[[CB2_PTR:.+]] = alloca i8
// LLVM:   %[[UII_PTR:.+]] = alloca i32
// LLVM:   %[[UIL_PTR:.+]] = alloca i64
// LLVM:   %[[UIF_PTR:.+]] = alloca float
// LLVM:   %[[UID_PTR:.+]] = alloca double
// LLVM:   %[[UIB_PTR:.+]] = alloca i8
// LLVM:   store i32 1, ptr %[[I_PTR]]
// LLVM:   store i64 2, ptr %[[L_PTR]]
// LLVM:   store float 3.000000e+00, ptr %[[F_PTR]]
// LLVM:   store double 4.000000e+00, ptr %[[D_PTR]]
// LLVM:   store i8 1, ptr %[[B1_PTR]]
// LLVM:   store i8 0, ptr %[[B2_PTR]]
// LLVM:   store i32 1, ptr %[[CI_PTR]]
// LLVM:   store i64 2, ptr %[[CL_PTR]]
// LLVM:   store float 3.000000e+00, ptr %[[CF_PTR]]
// LLVM:   store double 4.000000e+00, ptr %[[CD_PTR]]
// LLVM:   store i8 1, ptr %[[CB1_PTR]]
// LLVM:   store i8 0, ptr %[[CB2_PTR]]
// LLVM:   ret void

// OGCG: define dso_local void @_Z4testv()
// OGCG:   %[[I_PTR:.+]] = alloca i32
// OGCG:   %[[L_PTR:.+]] = alloca i64
// OGCG:   %[[F_PTR:.+]] = alloca float
// OGCG:   %[[D_PTR:.+]] = alloca double
// OGCG:   %[[B1_PTR:.+]] = alloca i8
// OGCG:   %[[B2_PTR:.+]] = alloca i8
// OGCG:   %[[CI_PTR:.+]] = alloca i32
// OGCG:   %[[CL_PTR:.+]] = alloca i64
// OGCG:   %[[CF_PTR:.+]] = alloca float
// OGCG:   %[[CD_PTR:.+]] = alloca double
// OGCG:   %[[CB1_PTR:.+]] = alloca i8
// OGCG:   %[[CB2_PTR:.+]] = alloca i8
// OGCG:   %[[UII_PTR:.+]] = alloca i32
// OGCG:   %[[UIL_PTR:.+]] = alloca i64
// OGCG:   %[[UIF_PTR:.+]] = alloca float
// OGCG:   %[[UID_PTR:.+]] = alloca double
// OGCG:   %[[UIB_PTR:.+]] = alloca i8
// OGCG:   store i32 1, ptr %[[I_PTR]]
// OGCG:   store i64 2, ptr %[[L_PTR]]
// OGCG:   store float 3.000000e+00, ptr %[[F_PTR]]
// OGCG:   store double 4.000000e+00, ptr %[[D_PTR]]
// OGCG:   store i8 1, ptr %[[B1_PTR]]
// OGCG:   store i8 0, ptr %[[B2_PTR]]
// OGCG:   store i32 1, ptr %[[CI_PTR]]
// OGCG:   store i64 2, ptr %[[CL_PTR]]
// OGCG:   store float 3.000000e+00, ptr %[[CF_PTR]]
// OGCG:   store double 4.000000e+00, ptr %[[CD_PTR]]
// OGCG:   store i8 1, ptr %[[CB1_PTR]]
// OGCG:   store i8 0, ptr %[[CB2_PTR]]
// OGCG:   ret void

void value_init() {
  float f{};
  bool b{};
  int i{};
}

// CIR: cir.func{{.*}} @_Z10value_initv()
// CIR:   %[[F_PTR:.+]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f", init]
// CIR:   %[[B_PTR:.+]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b", init]
// CIR:   %[[I_PTR:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR:   %[[ZEROF:.+]] = cir.const #cir.fp<0.000000e+00> : !cir.float
// CIR:   cir.store {{.*}} %[[ZEROF]], %[[F_PTR]] : !cir.float, !cir.ptr<!cir.float>
// CIR:   %[[FALSE:.+]] = cir.const #false
// CIR:   cir.store {{.*}} %[[FALSE]], %[[B_PTR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:   %[[ZEROI:.+]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store {{.*}} %[[ZEROI]], %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.return

// LLVM: define dso_local void @_Z10value_initv()
// LLVM:   %[[F_PTR:.+]] = alloca float
// LLVM:   %[[B_PTR:.+]] = alloca i8
// LLVM:   %[[I_PTR:.+]] = alloca i32
// LLVM:   store float 0.000000e+00, ptr %[[F_PTR]]
// LLVM:   store i8 0, ptr %[[B_PTR]]
// LLVM:   store i32 0, ptr %[[I_PTR]]
// LLVM:   ret void

// OGCG: define dso_local void @_Z10value_initv()
// OGCG:   %[[F_PTR:.+]] = alloca float
// OGCG:   %[[B_PTR:.+]] = alloca i8
// OGCG:   %[[I_PTR:.+]] = alloca i32
// OGCG:   store float 0.000000e+00, ptr %[[F_PTR]]
// OGCG:   store i8 0, ptr %[[B_PTR]]
// OGCG:   store i32 0, ptr %[[I_PTR]]
// OGCG:   ret void
