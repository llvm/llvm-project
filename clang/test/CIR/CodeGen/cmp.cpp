// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void c0(int a, int b) {
  bool x = a > b;
  x = a < b;
  x = a <= b;
  x = a >= b;
  x = a != b;
  x = a == b;
}

// CIR-LABEL: cir.func @_Z2c0ii(

// CIR: %[[A_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init]

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]]
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(gt, %[[A1]], %[[B1]]) : !s32i, !cir.bool
// CIR: cir.store{{.*}} {{.*}}, %[[X_PTR]]

// CIR: %[[A2:.*]] = cir.load{{.*}} %[[A_PTR]]
// CIR: %[[B2:.*]] = cir.load{{.*}} %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(lt, %[[A2]], %[[B2]]) : !s32i, !cir.bool

// CIR: %[[A3:.*]] = cir.load{{.*}} %[[A_PTR]]
// CIR: %[[B3:.*]] = cir.load{{.*}} %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(le, %[[A3]], %[[B3]]) : !s32i, !cir.bool

// CIR: %[[A4:.*]] = cir.load{{.*}} %[[A_PTR]]
// CIR: %[[B4:.*]] = cir.load{{.*}} %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(ge, %[[A4]], %[[B4]]) : !s32i, !cir.bool

// CIR: %[[A5:.*]] = cir.load{{.*}} %[[A_PTR]]
// CIR: %[[B5:.*]] = cir.load{{.*}} %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(ne, %[[A5]], %[[B5]]) : !s32i, !cir.bool

// CIR: %[[A6:.*]] = cir.load{{.*}} %[[A_PTR]]
// CIR: %[[B6:.*]] = cir.load{{.*}} %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(eq, %[[A6]], %[[B6]]) : !s32i, !cir.bool

// LLVM-LABEL: define void @_Z2c0ii(i32 %0, i32 %1) {
// LLVM: %[[PTR1:.*]] = alloca i32, i64 1
// LLVM: %[[PTR2:.*]] = alloca i32, i64 1
// LLVM: %[[BOOL_PTR:.*]] = alloca i8, i64 1
// LLVM: store i32 %0, ptr %[[PTR1]]
// LLVM: store i32 %1, ptr %[[PTR2]]

// LLVM: %[[A1:.*]] = load i32, ptr %[[PTR1]]
// LLVM: %[[B1:.*]] = load i32, ptr %[[PTR2]]
// LLVM: %[[CMP1:.*]] = icmp sgt i32 %[[A1]], %[[B1]]
// LLVM: %[[ZEXT1:.*]] = zext i1 %[[CMP1]] to i8
// LLVM: store i8 %[[ZEXT1]], ptr %[[BOOL_PTR]]

// LLVM: %[[A2:.*]] = load i32, ptr %[[PTR1]]
// LLVM: %[[B2:.*]] = load i32, ptr %[[PTR2]]
// LLVM: %[[CMP2:.*]] = icmp slt i32 %[[A2]], %[[B2]]
// LLVM: %[[ZEXT2:.*]] = zext i1 %[[CMP2]] to i8
// LLVM: store i8 %[[ZEXT2]], ptr %[[BOOL_PTR]]

// LLVM: %[[A3:.*]] = load i32, ptr %[[PTR1]]
// LLVM: %[[B3:.*]] = load i32, ptr %[[PTR2]]
// LLVM: %[[CMP3:.*]] = icmp sle i32 %[[A3]], %[[B3]]
// LLVM: %[[ZEXT3:.*]] = zext i1 %[[CMP3]] to i8
// LLVM: store i8 %[[ZEXT3]], ptr %[[BOOL_PTR]]

// LLVM: %[[A4:.*]] = load i32, ptr %[[PTR1]]
// LLVM: %[[B4:.*]] = load i32, ptr %[[PTR2]]
// LLVM: %[[CMP4:.*]] = icmp sge i32 %[[A4]], %[[B4]]
// LLVM: %[[ZEXT4:.*]] = zext i1 %[[CMP4]] to i8
// LLVM: store i8 %[[ZEXT4]], ptr %[[BOOL_PTR]]

// LLVM: %[[A5:.*]] = load i32, ptr %[[PTR1]]
// LLVM: %[[B5:.*]] = load i32, ptr %[[PTR2]]
// LLVM: %[[CMP5:.*]] = icmp ne i32 %[[A5]], %[[B5]]
// LLVM: %[[ZEXT5:.*]] = zext i1 %[[CMP5]] to i8
// LLVM: store i8 %[[ZEXT5]], ptr %[[BOOL_PTR]]

// LLVM: %[[A6:.*]] = load i32, ptr %[[PTR1]]
// LLVM: %[[B6:.*]] = load i32, ptr %[[PTR2]]
// LLVM: %[[CMP6:.*]] = icmp eq i32 %[[A6]], %[[B6]]
// LLVM: %[[ZEXT6:.*]] = zext i1 %[[CMP6]] to i8
// LLVM: store i8 %[[ZEXT6]], ptr %[[BOOL_PTR]]

// OGCG-LABEL: define dso_local void @_Z2c0ii(i32 {{.*}} %a, i32 {{.*}} %b) {{.*}} {
// OGCG: %[[PTR1:.*]] = alloca i32
// OGCG: %[[PTR2:.*]] = alloca i32
// OGCG: %[[BOOL_PTR:.*]] = alloca i8
// OGCG: store i32 %a, ptr %[[PTR1]]
// OGCG: store i32 %b, ptr %[[PTR2]]

// OGCO: %[[A1:.*]] = load i32, ptr %[[PTR1]]
// OGCO: %[[B1:.*]] = load i32, ptr %[[PTR2]]
// OGCO: %[[CMP1:.*]] = icmp sgt i32 %[[A1]], %[[B1]]
// OGCO: %[[ZEXT1:.*]] = zext i1 %[[CMP1]] to i8
// OGCO: store i8 %[[ZEXT1]], ptr %[[BOOL_PTR]]

// OGCO: %[[A2:.*]] = load i32, ptr %[[PTR1]]
// OGCO: %[[B2:.*]] = load i32, ptr %[[PTR2]]
// OGCO: %[[CMP2:.*]] = icmp slt i32 %[[A2]], %[[B2]]
// OGCO: %[[ZEXT2:.*]] = zext i1 %[[CMP2]] to i8
// OGCO: store i8 %[[ZEXT2]], ptr %[[BOOL_PTR]]

// OGCO: %[[A3:.*]] = load i32, ptr %[[PTR1]]
// OGCO: %[[B3:.*]] = load i32, ptr %[[PTR2]]
// OGCO: %[[CMP3:.*]] = icmp sle i32 %[[A3]], %[[B3]]
// OGCO: %[[ZEXT3:.*]] = zext i1 %[[CMP3]] to i8
// OGCO: store i8 %[[ZEXT3]], ptr %[[BOOL_PTR]]

// OGCO: %[[A4:.*]] = load i32, ptr %[[PTR1]]
// OGCO: %[[B4:.*]] = load i32, ptr %[[PTR2]]
// OGCO: %[[CMP4:.*]] = icmp sge i32 %[[A4]], %[[B4]]
// OGCO: %[[ZEXT4:.*]] = zext i1 %[[CMP4]] to i8
// OGCO: store i8 %[[ZEXT4]], ptr %[[BOOL_PTR]]

// OGCO: %[[A5:.*]] = load i32, ptr %[[PTR1]]
// OGCO: %[[B5:.*]] = load i32, ptr %[[PTR2]]
// OGCO: %[[CMP5:.*]] = icmp ne i32 %[[A5]], %[[B5]]
// OGCO: %[[ZEXT5:.*]] = zext i1 %[[CMP5]] to i8
// OGCO: store i8 %[[ZEXT5]], ptr %[[BOOL_PTR]]

// OGCO: %[[A6:.*]] = load i32, ptr %[[PTR1]]
// OGCO: %[[B6:.*]] = load i32, ptr %[[PTR2]]
// OGCO: %[[CMP6:.*]] = icmp eq i32 %[[A6]], %[[B6]]
// OGCO: %[[ZEXT6:.*]] = zext i1 %[[CMP6]] to i8
// OGCO: store i8 %[[ZEXT6]], ptr %[[BOOL_PTR]]

void c0_unsigned(unsigned int a, unsigned int b) {
  bool x = a > b;
  x = a < b;
  x = a <= b;
  x = a >= b;
  x = a != b;
  x = a == b;
}

// CIR-LABEL: cir.func @_Z11c0_unsignedjj(

// CIR: %[[U_A_PTR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CIR: %[[U_B_PTR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["b", init]
// CIR: %[[U_X_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init]

// CIR: %[[UA1:.*]] = cir.load{{.*}} %[[U_A_PTR]]
// CIR: %[[UB1:.*]] = cir.load{{.*}} %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(gt, %[[UA1]], %[[UB1]]) : !u32i, !cir.bool

// CIR: %[[UA2:.*]] = cir.load{{.*}} %[[U_A_PTR]]
// CIR: %[[UB2:.*]] = cir.load{{.*}} %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(lt, %[[UA2]], %[[UB2]]) : !u32i, !cir.bool

// CIR: %[[UA3:.*]] = cir.load{{.*}} %[[U_A_PTR]]
// CIR: %[[UB3:.*]] = cir.load{{.*}} %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(le, %[[UA3]], %[[UB3]]) : !u32i, !cir.bool

// CIR: %[[UA4:.*]] = cir.load{{.*}} %[[U_A_PTR]]
// CIR: %[[UB4:.*]] = cir.load{{.*}} %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(ge, %[[UA4]], %[[UB4]]) : !u32i, !cir.bool

// CIR: %[[UA5:.*]] = cir.load{{.*}} %[[U_A_PTR]]
// CIR: %[[UB5:.*]] = cir.load{{.*}} %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(ne, %[[UA5]], %[[UB5]]) : !u32i, !cir.bool

// CIR: %[[UA6:.*]] = cir.load{{.*}} %[[U_A_PTR]]
// CIR: %[[UB6:.*]] = cir.load{{.*}} %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(eq, %[[UA6]], %[[UB6]]) : !u32i, !cir.bool

// LLVM-LABEL: define void @_Z11c0_unsignedjj(i32 %0, i32 %1) {
// LLVM: %[[U_PTR1:.*]] = alloca i32, i64 1
// LLVM: %[[U_PTR2:.*]] = alloca i32, i64 1
// LLVM: %[[U_BOOL_PTR:.*]] = alloca i8, i64 1
// LLVM: store i32 %0, ptr %[[U_PTR1]]
// LLVM: store i32 %1, ptr %[[U_PTR2]]

// LLVM: %[[UA1:.*]] = load i32, ptr %[[U_PTR1]]
// LLVM: %[[UB1:.*]] = load i32, ptr %[[U_PTR2]]
// LLVM: %[[UCMP1:.*]] = icmp ugt i32 %[[UA1]], %[[UB1]]
// LLVM: %[[UZEXT1:.*]] = zext i1 %[[UCMP1]] to i8
// LLVM: store i8 %[[UZEXT1]], ptr %[[U_BOOL_PTR]]

// LLVM: %[[UA2:.*]] = load i32, ptr %[[U_PTR1]]
// LLVM: %[[UB2:.*]] = load i32, ptr %[[U_PTR2]]
// LLVM: %[[UCMP2:.*]] = icmp ult i32 %[[UA2]], %[[UB2]]
// LLVM: %[[UZEXT2:.*]] = zext i1 %[[UCMP2]] to i8
// LLVM: store i8 %[[UZEXT2]], ptr %[[U_BOOL_PTR]]

// LLVM: %[[UA3:.*]] = load i32, ptr %[[U_PTR1]]
// LLVM: %[[UB3:.*]] = load i32, ptr %[[U_PTR2]]
// LLVM: %[[UCMP3:.*]] = icmp ule i32 %[[UA3]], %[[UB3]]
// LLVM: %[[UZEXT3:.*]] = zext i1 %[[UCMP3]] to i8
// LLVM: store i8 %[[UZEXT3]], ptr %[[U_BOOL_PTR]]

// LLVM: %[[UA4:.*]] = load i32, ptr %[[U_PTR1]]
// LLVM: %[[UB4:.*]] = load i32, ptr %[[U_PTR2]]
// LLVM: %[[UCMP4:.*]] = icmp uge i32 %[[UA4]], %[[UB4]]
// LLVM: %[[UZEXT4:.*]] = zext i1 %[[UCMP4]] to i8
// LLVM: store i8 %[[UZEXT4]], ptr %[[U_BOOL_PTR]]

// LLVM: %[[UA5:.*]] = load i32, ptr %[[U_PTR1]]
// LLVM: %[[UB5:.*]] = load i32, ptr %[[U_PTR2]]
// LLVM: %[[UCMP5:.*]] = icmp ne i32 %[[UA5]], %[[UB5]]
// LLVM: %[[UZEXT5:.*]] = zext i1 %[[UCMP5]] to i8
// LLVM: store i8 %[[UZEXT5]], ptr %[[U_BOOL_PTR]]

// LLVM: %[[UA6:.*]] = load i32, ptr %[[U_PTR1]]
// LLVM: %[[UB6:.*]] = load i32, ptr %[[U_PTR2]]
// LLVM: %[[UCMP6:.*]] = icmp eq i32 %[[UA6]], %[[UB6]]
// LLVM: %[[UZEXT6:.*]] = zext i1 %[[UCMP6]] to i8
// LLVM: store i8 %[[UZEXT6]], ptr %[[U_BOOL_PTR]]

// OGCG-LABEL: define dso_local void @_Z11c0_unsignedjj(i32 {{.*}} %a, i32 {{.*}} %b) {{.*}} {
// OGCG: %[[U_PTR1:.*]] = alloca i32
// OGCG: %[[U_PTR2:.*]] = alloca i32
// OGCG: %[[U_BOOL_PTR:.*]] = alloca i8
// OGCG: store i32 %a, ptr %[[U_PTR1]]
// OGCG: store i32 %b, ptr %[[U_PTR2]]

// OGCG: %[[UA1:.*]] = load i32, ptr %[[U_PTR1]]
// OGCG: %[[UB1:.*]] = load i32, ptr %[[U_PTR2]]
// OGCG: %[[UCMP1:.*]] = icmp ugt i32 %[[UA1]], %[[UB1]]
// OGCG: %[[UZEXT1:.*]] = zext i1 %[[UCMP1]] to i8
// OGCG: store i8 %[[UZEXT1]], ptr %[[U_BOOL_PTR]]

// OGCG: %[[UA2:.*]] = load i32, ptr %[[U_PTR1]]
// OGCG: %[[UB2:.*]] = load i32, ptr %[[U_PTR2]]
// OGCG: %[[UCMP2:.*]] = icmp ult i32 %[[UA2]], %[[UB2]]
// OGCG: %[[UZEXT2:.*]] = zext i1 %[[UCMP2]] to i8
// OGCG: store i8 %[[UZEXT2]], ptr %[[U_BOOL_PTR]]

// OGCG: %[[UA3:.*]] = load i32, ptr %[[U_PTR1]]
// OGCG: %[[UB3:.*]] = load i32, ptr %[[U_PTR2]]
// OGCG: %[[UCMP3:.*]] = icmp ule i32 %[[UA3]], %[[UB3]]
// OGCG: %[[UZEXT3:.*]] = zext i1 %[[UCMP3]] to i8
// OGCG: store i8 %[[UZEXT3]], ptr %[[U_BOOL_PTR]]

// OGCG: %[[UA4:.*]] = load i32, ptr %[[U_PTR1]]
// OGCG: %[[UB4:.*]] = load i32, ptr %[[U_PTR2]]
// OGCG: %[[UCMP4:.*]] = icmp uge i32 %[[UA4]], %[[UB4]]
// OGCG: %[[UZEXT4:.*]] = zext i1 %[[UCMP4]] to i8
// OGCG: store i8 %[[UZEXT4]], ptr %[[U_BOOL_PTR]]

// OGCG: %[[UA5:.*]] = load i32, ptr %[[U_PTR1]]
// OGCG: %[[UB5:.*]] = load i32, ptr %[[U_PTR2]]
// OGCG: %[[UCMP5:.*]] = icmp ne i32 %[[UA5]], %[[UB5]]
// OGCG: %[[UZEXT5:.*]] = zext i1 %[[UCMP5]] to i8
// OGCG: store i8 %[[UZEXT5]], ptr %[[U_BOOL_PTR]]

// OGCG: %[[UA6:.*]] = load i32, ptr %[[U_PTR1]]
// OGCG: %[[UB6:.*]] = load i32, ptr %[[U_PTR2]]
// OGCG: %[[UCMP6:.*]] = icmp eq i32 %[[UA6]], %[[UB6]]
// OGCG: %[[UZEXT6:.*]] = zext i1 %[[UCMP6]] to i8
// OGCG: store i8 %[[UZEXT6]], ptr %[[U_BOOL_PTR]]

void c0_float(float a, float b) {
  bool x = a > b;
  x = a < b;
  x = a <= b;
  x = a >= b;
  x = a != b;
  x = a == b;
}

// CIR-LABEL: cir.func @_Z8c0_floatff(%arg0: !cir.float{{.*}}, %arg1: !cir.float{{.*}}) {
// CIR: %[[A_PTR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init]

// CIR: cir.store{{.*}} %arg0, %[[A_PTR]] : !cir.float, !cir.ptr<!cir.float>
// CIR: cir.store{{.*}} %arg1, %[[B_PTR]] : !cir.float, !cir.ptr<!cir.float>

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP1:.*]] = cir.cmp(gt, %[[A1]], %[[B1]]) : !cir.float, !cir.bool
// CIR: cir.store{{.*}} %[[CMP1]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR: %[[A2:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B2:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP2:.*]] = cir.cmp(lt, %[[A2]], %[[B2]]) : !cir.float, !cir.bool
// CIR: cir.store{{.*}} %[[CMP2]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR: %[[A3:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B3:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP3:.*]] = cir.cmp(le, %[[A3]], %[[B3]]) : !cir.float, !cir.bool
// CIR: cir.store{{.*}} %[[CMP3]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR: %[[A4:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B4:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP4:.*]] = cir.cmp(ge, %[[A4]], %[[B4]]) : !cir.float, !cir.bool
// CIR: cir.store{{.*}} %[[CMP4]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR: %[[A5:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B5:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP5:.*]] = cir.cmp(ne, %[[A5]], %[[B5]]) : !cir.float, !cir.bool
// CIR: cir.store{{.*}} %[[CMP5]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR: %[[A6:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B6:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP6:.*]] = cir.cmp(eq, %[[A6]], %[[B6]]) : !cir.float, !cir.bool
// CIR: cir.store{{.*}} %[[CMP6]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM-LABEL: define void @_Z8c0_floatff(float %0, float %1) {
// LLVM: %[[A_PTR:.*]] = alloca float
// LLVM: %[[B_PTR:.*]] = alloca float
// LLVM: store float %0, ptr %[[A_PTR]]
// LLVM: store float %1, ptr %[[B_PTR]]

// LLVM: load float, ptr %[[A_PTR]]
// LLVM: load float, ptr %[[B_PTR]]
// LLVM: fcmp ogt float %{{.*}}, %{{.*}}
// LLVM: zext i1 %{{.*}} to i8

// LLVM: fcmp olt float %{{.*}}, %{{.*}}
// LLVM: fcmp ole float %{{.*}}, %{{.*}}
// LLVM: fcmp oge float %{{.*}}, %{{.*}}
// LLVM: fcmp une float %{{.*}}, %{{.*}}
// LLVM: fcmp oeq float %{{.*}}, %{{.*}}

// OGCG-LABEL: define dso_local void @_Z8c0_floatff(float {{.*}} %a, float  {{.*}} %b)  {{.*}} {
// OGCG: %[[A_PTR:.*]] = alloca float
// OGCG: %[[B_PTR:.*]] = alloca float
// OGCG: store float %a, ptr %[[A_PTR]]
// OGCG: store float %b, ptr %[[B_PTR]]

// OGCG: load float, ptr %[[A_PTR]]
// OGCG: load float, ptr %[[B_PTR]]
// OGCG: fcmp ogt float %{{.*}}, %{{.*}}
// OGCG: zext i1 %{{.*}} to i8

// OGCG: fcmp olt float %{{.*}}, %{{.*}}
// OGCG: fcmp ole float %{{.*}}, %{{.*}}
// OGCG: fcmp oge float %{{.*}}, %{{.*}}
// OGCG: fcmp une float %{{.*}}, %{{.*}}
// OGCG: fcmp oeq float %{{.*}}, %{{.*}}

void pointer_cmp(int *a, int *b) {
  bool x = a > b;
  x = a < b;
  x = a >= b;
  x = a <= b;
  x = a == b;
  x = a != b;
}

// CIR-LABEL: cir.func @_Z11pointer_cmpPiS_(%arg0: !cir.ptr<!s32i>{{.*}}, %arg1: !cir.ptr<!s32i>{{.*}}) {
// CIR: %[[A_PTR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["b", init]

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR: %{{.*}} = cir.cmp(gt, %[[A1]], %[[B1]]) : !cir.ptr<!s32i>, !cir.bool

// CIR: cir.cmp(lt, {{.*}}, {{.*}}) : !cir.ptr<!s32i>, !cir.bool
// CIR: cir.cmp(ge, {{.*}}, {{.*}}) : !cir.ptr<!s32i>, !cir.bool
// CIR: cir.cmp(le, {{.*}}, {{.*}}) : !cir.ptr<!s32i>, !cir.bool
// CIR: cir.cmp(eq, {{.*}}, {{.*}}) : !cir.ptr<!s32i>, !cir.bool
// CIR: cir.cmp(ne, {{.*}}, {{.*}}) : !cir.ptr<!s32i>, !cir.bool

// LLVM-LABEL: define void @_Z11pointer_cmpPiS_(ptr %0, ptr %1) {
// LLVM: %[[A_PTR:.*]] = alloca ptr
// LLVM: %[[B_PTR:.*]] = alloca ptr
// LLVM: store ptr %0, ptr %[[A_PTR]]
// LLVM: store ptr %1, ptr %[[B_PTR]]

// LLVM: load ptr, ptr %[[A_PTR]]
// LLVM: load ptr, ptr %[[B_PTR]]
// LLVM: icmp ugt ptr %{{.*}}, %{{.*}}
// LLVM: zext i1 %{{.*}} to i8
// LLVM: icmp ult ptr %{{.*}}, %{{.*}}
// LLVM: icmp uge ptr %{{.*}}, %{{.*}}
// LLVM: icmp ule ptr %{{.*}}, %{{.*}}
// LLVM: icmp eq ptr %{{.*}}, %{{.*}}
// LLVM: icmp ne ptr %{{.*}}, %{{.*}}

// OGCG-LABEL: define dso_local void @_Z11pointer_cmpPiS_(ptr {{.*}} %a, ptr {{.*}} %b) {{.*}} {
// OGCG: %[[A_PTR:.*]] = alloca ptr
// OGCG: %[[B_PTR:.*]] = alloca ptr
// OGCG: store ptr %a, ptr %[[A_PTR]]
// OGCG: store ptr %b, ptr %[[B_PTR]]

// OGCG: load ptr, ptr %[[A_PTR]]
// OGCG: load ptr, ptr %[[B_PTR]]
// OGCG: icmp ugt ptr %{{.*}}, %{{.*}}
// OGCG: zext i1 %{{.*}} to i8
// OGCG: icmp ult ptr %{{.*}}, %{{.*}}
// OGCG: icmp uge ptr %{{.*}}, %{{.*}}
// OGCG: icmp ule ptr %{{.*}}, %{{.*}}
// OGCG: icmp eq ptr %{{.*}}, %{{.*}}
// OGCG: icmp ne ptr %{{.*}}, %{{.*}}

void bool_cmp(bool a, bool b) {
  bool x = a > b;
  x = a < b;
  x = a >= b;
  x = a <= b;
  x = a == b;
  x = a != b;
}

// CIR-LABEL: cir.func @_Z8bool_cmpbb(%arg0: !cir.bool{{.*}}, %arg1: !cir.bool{{.*}}) {
// CIR: %[[A_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init]

// CIR: %[[A1:.*]] = cir.load{{.*}} %[[A_PTR]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[A1_INT:.*]] = cir.cast(bool_to_int, %[[A1]] : !cir.bool), !s32i
// CIR: %[[B1:.*]] = cir.load{{.*}} %[[B_PTR]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[B1_INT:.*]] = cir.cast(bool_to_int, %[[B1]] : !cir.bool), !s32i
// CIR: %{{.*}} = cir.cmp(gt, %[[A1_INT]], %[[B1_INT]]) : !s32i, !cir.bool
// CIR: cir.store{{.*}} {{.*}}, %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR: cir.cmp(lt
// CIR: cir.cmp(ge
// CIR: cir.cmp(le
// CIR: cir.cmp(eq
// CIR: cir.cmp(ne

// LLVM-LABEL: define void @_Z8bool_cmpbb(i1 %0, i1 %1) {
// LLVM: %[[A_PTR:.*]] = alloca i8
// LLVM: %[[B_PTR:.*]] = alloca i8
// LLVM: %[[X_PTR:.*]] = alloca i8
// LLVM: %[[A_INIT:.*]] = zext i1 %0 to i8
// LLVM: store i8 %[[A_INIT]], ptr %[[A_PTR]]
// LLVM: %[[B_INIT:.*]] = zext i1 %1 to i8
// LLVM: store i8 %[[B_INIT]], ptr %[[B_PTR]]

// LLVM: %[[A1:.*]] = load i8, ptr %[[A_PTR]]
// LLVM: %[[A1_TRUNC:.*]] = trunc i8 %[[A1]] to i1
// LLVM: %[[A1_EXT:.*]] = zext i1 %[[A1_TRUNC]] to i32
// LLVM: %[[B1:.*]] = load i8, ptr %[[B_PTR]]
// LLVM: %[[B1_TRUNC:.*]] = trunc i8 %[[B1]] to i1
// LLVM: %[[B1_EXT:.*]] = zext i1 %[[B1_TRUNC]] to i32
// LLVM: %[[CMP1:.*]] = icmp sgt i32 %[[A1_EXT]], %[[B1_EXT]]
// LLVM: %[[CMP1_BOOL:.*]] = zext i1 %[[CMP1]] to i8
// LLVM: store i8 %[[CMP1_BOOL]], ptr %[[X_PTR]]

// LLVM: icmp slt
// LLVM: icmp sge
// LLVM: icmp sle
// LLVM: icmp eq
// LLVM: icmp ne

// OGCG-LABEL: define dso_local void @_Z8bool_cmpbb(i1 {{.*}} %a, i1 {{.*}} %b) {{.*}} {
// OGCG: %[[A_PTR:.*]] = alloca i8
// OGCG: %[[B_PTR:.*]] = alloca i8
// OGCG: %[[X_PTR:.*]] = alloca i8
// OGCG: %[[A_INIT:.*]] = zext i1 %a to i8
// OGCG: store i8 %[[A_INIT]], ptr %[[A_PTR]]
// OGCG: %[[B_INIT:.*]] = zext i1 %b to i8
// OGCG: store i8 %[[B_INIT]], ptr %[[B_PTR]]

// OGCG: %[[A1:.*]] = load i8, ptr %[[A_PTR]]
// OGCG: %[[A1_TRUNC:.*]] = trunc i8 %[[A1]] to i1
// OGCG: %[[A1_EXT:.*]] = zext i1 %[[A1_TRUNC]] to i32
// OGCG: %[[B1:.*]] = load i8, ptr %[[B_PTR]]
// OGCG: %[[B1_TRUNC:.*]] = trunc i8 %[[B1]] to i1
// OGCG: %[[B1_EXT:.*]] = zext i1 %[[B1_TRUNC]] to i32
// OGCG: %[[CMP1:.*]] = icmp sgt i32 %[[A1_EXT]], %[[B1_EXT]]
// OGCG: %[[CMP1_BOOL:.*]] = zext i1 %[[CMP1]] to i8
// OGCG: store i8 %[[CMP1_BOOL]], ptr %[[X_PTR]]

// OGCG: icmp slt
// OGCG: icmp sge
// OGCG: icmp sle
// OGCG: icmp eq
// OGCG: icmp ne
