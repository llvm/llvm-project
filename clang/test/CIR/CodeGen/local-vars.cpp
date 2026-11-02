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
// CIR:    %[[I_PTR:.*]] = cir.alloca "i" align(4) init : !cir.ptr<!s32i>
// CIR:    %[[L_PTR:.*]] = cir.alloca "l" align(8) init : !cir.ptr<!s64i>
// CIR:    %[[F_PTR:.*]] = cir.alloca "f" align(4) init : !cir.ptr<!cir.float>
// CIR:    %[[D_PTR:.*]] = cir.alloca "d" align(8) init : !cir.ptr<!cir.double>
// CIR:    %[[B1_PTR:.*]] = cir.alloca "b1" align(1) init : !cir.ptr<!cir.bool>
// CIR:    %[[B2_PTR:.*]] = cir.alloca "b2" align(1) init : !cir.ptr<!cir.bool>
// CIR:    %[[CI_PTR:.*]] = cir.alloca "ci" align(4) init const : !cir.ptr<!s32i>
// CIR:    %[[CL_PTR:.*]] = cir.alloca "cl" align(8) init const : !cir.ptr<!s64i>
// CIR:    %[[CF_PTR:.*]] = cir.alloca "cf" align(4) init const : !cir.ptr<!cir.float>
// CIR:    %[[CD_PTR:.*]] = cir.alloca "cd" align(8) init const : !cir.ptr<!cir.double>
// CIR:    %[[CB1_PTR:.*]] = cir.alloca "cb1" align(1) init const : !cir.ptr<!cir.bool>
// CIR:    %[[CB2_PTR:.*]] = cir.alloca "cb2" align(1) init const : !cir.ptr<!cir.bool>
// CIR:    %[[UII_PTR:.*]] = cir.alloca "uii" align(4) : !cir.ptr<!s32i>
// CIR:    %[[UIL_PTR:.*]] = cir.alloca "uil" align(8) : !cir.ptr<!s64i>
// CIR:    %[[UIF_PTR:.*]] = cir.alloca "uif" align(4) : !cir.ptr<!cir.float>
// CIR:    %[[UID_PTR:.*]] = cir.alloca "uid" align(8) : !cir.ptr<!cir.double>
// CIR:    %[[UIB_PTR:.*]] = cir.alloca "uib" align(1) : !cir.ptr<!cir.bool>
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

  float f2 = {};
  bool b2 = {};
  int i2 = {};

  bool scalar_value_init_expr = int() == 0;
}

// CIR: cir.func{{.*}} @_Z10value_initv()
// CIR:   %[[F_PTR:.+]] = cir.alloca "f" {{.*}} init : !cir.ptr<!cir.float>
// CIR:   %[[B_PTR:.+]] = cir.alloca "b" {{.*}} init : !cir.ptr<!cir.bool>
// CIR:   %[[I_PTR:.+]] = cir.alloca "i" {{.*}} init : !cir.ptr<!s32i>
// CIR:   %[[F2_PTR:.+]] = cir.alloca "f2" {{.*}} init : !cir.ptr<!cir.float>
// CIR:   %[[B2_PTR:.+]] = cir.alloca "b2" {{.*}} init : !cir.ptr<!cir.bool>
// CIR:   %[[I2_PTR:.+]] = cir.alloca "i2" {{.*}} init : !cir.ptr<!s32i>
// CIR:   %[[S_PTR:.+]] = cir.alloca "scalar_value_init_expr" {{.*}} init : !cir.ptr<!cir.bool>
// CIR:   %[[ZEROF1:.+]] = cir.const #cir.fp<0.000000e+00> : !cir.float
// CIR:   cir.store{{.*}} %[[ZEROF1]], %[[F_PTR]] : !cir.float, !cir.ptr<!cir.float>
// CIR:   %[[FALSE1:.+]] = cir.const #false
// CIR:   cir.store{{.*}} %[[FALSE1]], %[[B_PTR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:   %[[ZEROI1:.+]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZEROI1]], %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[ZEROF2:.+]] = cir.const #cir.fp<0.000000e+00> : !cir.float
// CIR:   cir.store{{.*}} %[[ZEROF2]], %[[F2_PTR]] : !cir.float, !cir.ptr<!cir.float>
// CIR:   %[[FALSE2:.+]] = cir.const #false
// CIR:   cir.store{{.*}} %[[FALSE2]], %[[B2_PTR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:   %[[ZEROI2:.+]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZEROI2]], %[[I2_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[ZEROI_LHS:.+]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[ZEROI_RHS:.+]] = cir.const #cir.int<0> : !s32i
// CIR:   %[[CMP:.+]] = cir.cmp eq %[[ZEROI_LHS]], %[[ZEROI_RHS]] : !s32i
// CIR:   cir.store{{.*}} %[[CMP]], %[[S_PTR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z10value_initv()
// LLVM:   %[[F_PTR:.+]] = alloca float
// LLVM:   %[[B_PTR:.+]] = alloca i8
// LLVM:   %[[I_PTR:.+]] = alloca i32
// LLVM:   %[[F2_PTR:.+]] = alloca float
// LLVM:   %[[B2_PTR:.+]] = alloca i8
// LLVM:   %[[I2_PTR:.+]] = alloca i32
// LLVM:   %[[S_PTR:.+]] = alloca i8
// LLVM:   store float 0.000000e+00, ptr %[[F_PTR]]
// LLVM:   store i8 0, ptr %[[B_PTR]]
// LLVM:   store i32 0, ptr %[[I_PTR]]
// LLVM:   store float 0.000000e+00, ptr %[[F2_PTR]]
// LLVM:   store i8 0, ptr %[[B2_PTR]]
// LLVM:   store i32 0, ptr %[[I2_PTR]]
// LLVM:   store i8 1, ptr %[[S_PTR]]
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z10value_initv()
// OGCG:   %[[F_PTR:.+]] = alloca float
// OGCG:   %[[B_PTR:.+]] = alloca i8
// OGCG:   %[[I_PTR:.+]] = alloca i32
// OGCG:   %[[F2_PTR:.+]] = alloca float
// OGCG:   %[[B2_PTR:.+]] = alloca i8
// OGCG:   %[[I2_PTR:.+]] = alloca i32
// OGCG:   %[[S_PTR:.+]] = alloca i8
// OGCG:   store float 0.000000e+00, ptr %[[F_PTR]]
// OGCG:   store i8 0, ptr %[[B_PTR]]
// OGCG:   store i32 0, ptr %[[I_PTR]]
// OGCG:   store float 0.000000e+00, ptr %[[F2_PTR]]
// OGCG:   store i8 0, ptr %[[B2_PTR]]
// OGCG:   store i32 0, ptr %[[I2_PTR]]
// OGCG:   store i8 1, ptr %[[S_PTR]]
// OGCG:   ret void
