// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -DCIR_ONLY %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

void c0(int a, int b) {
  bool x = a > b;
  x = a < b;
  x = a <= b;
  x = a >= b;
  x = a != b;
  x = a == b;
}

// CIR: cir.func @c0(

// CIR: %[[A_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init]

// CIR: %[[A1:.*]] = cir.load %[[A_PTR]]
// CIR: %[[B1:.*]] = cir.load %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(gt, %[[A1]], %[[B1]]) : !s32i, !cir.bool
// CIR: cir.store {{.*}}, %[[X_PTR]]

// CIR: %[[A2:.*]] = cir.load %[[A_PTR]]
// CIR: %[[B2:.*]] = cir.load %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(lt, %[[A2]], %[[B2]]) : !s32i, !cir.bool

// CIR: %[[A3:.*]] = cir.load %[[A_PTR]]
// CIR: %[[B3:.*]] = cir.load %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(le, %[[A3]], %[[B3]]) : !s32i, !cir.bool

// CIR: %[[A4:.*]] = cir.load %[[A_PTR]]
// CIR: %[[B4:.*]] = cir.load %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(ge, %[[A4]], %[[B4]]) : !s32i, !cir.bool

// CIR: %[[A5:.*]] = cir.load %[[A_PTR]]
// CIR: %[[B5:.*]] = cir.load %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(ne, %[[A5]], %[[B5]]) : !s32i, !cir.bool

// CIR: %[[A6:.*]] = cir.load %[[A_PTR]]
// CIR: %[[B6:.*]] = cir.load %[[B_PTR]]
// CIR: %{{.*}} = cir.cmp(eq, %[[A6]], %[[B6]]) : !s32i, !cir.bool

// LLVM: define void @c0(i32 %0, i32 %1) {
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

void c0_unsigned(unsigned int a, unsigned int b) {
  bool x = a > b;
  x = a < b;
  x = a <= b;
  x = a >= b;
  x = a != b;
  x = a == b;
}

// CIR: cir.func @c0_unsigned(

// CIR: %[[U_A_PTR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CIR: %[[U_B_PTR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["b", init]
// CIR: %[[U_X_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init]

// CIR: %[[UA1:.*]] = cir.load %[[U_A_PTR]]
// CIR: %[[UB1:.*]] = cir.load %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(gt, %[[UA1]], %[[UB1]]) : !u32i, !cir.bool

// CIR: %[[UA2:.*]] = cir.load %[[U_A_PTR]]
// CIR: %[[UB2:.*]] = cir.load %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(lt, %[[UA2]], %[[UB2]]) : !u32i, !cir.bool

// CIR: %[[UA3:.*]] = cir.load %[[U_A_PTR]]
// CIR: %[[UB3:.*]] = cir.load %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(le, %[[UA3]], %[[UB3]]) : !u32i, !cir.bool

// CIR: %[[UA4:.*]] = cir.load %[[U_A_PTR]]
// CIR: %[[UB4:.*]] = cir.load %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(ge, %[[UA4]], %[[UB4]]) : !u32i, !cir.bool

// CIR: %[[UA5:.*]] = cir.load %[[U_A_PTR]]
// CIR: %[[UB5:.*]] = cir.load %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(ne, %[[UA5]], %[[UB5]]) : !u32i, !cir.bool

// CIR: %[[UA6:.*]] = cir.load %[[U_A_PTR]]
// CIR: %[[UB6:.*]] = cir.load %[[U_B_PTR]]
// CIR: %{{.*}} = cir.cmp(eq, %[[UA6]], %[[UB6]]) : !u32i, !cir.bool

// LLVM: define void @c0_unsigned(i32 %0, i32 %1) {
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

void c0_float(float a, float b) {
  bool x = a > b;
  x = a < b;
  x = a <= b;
  x = a >= b;
  x = a != b;
  x = a == b;
}

// CIR: cir.func @c0_float(%arg0: !cir.float{{.*}}, %arg1: !cir.float{{.*}}) {
// CIR: %[[A_PTR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CIR: %[[B_PTR:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CIR: %[[X_PTR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["x", init]

// CIR: cir.store %arg0, %[[A_PTR]] : !cir.float, !cir.ptr<!cir.float>
// CIR: cir.store %arg1, %[[B_PTR]] : !cir.float, !cir.ptr<!cir.float>

// CIR: %[[A1:.*]] = cir.load %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B1:.*]] = cir.load %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP1:.*]] = cir.cmp(gt, %[[A1]], %[[B1]]) : !cir.float, !cir.bool
// CIR: cir.store %[[CMP1]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR: %[[A2:.*]] = cir.load %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B2:.*]] = cir.load %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP2:.*]] = cir.cmp(lt, %[[A2]], %[[B2]]) : !cir.float, !cir.bool
// CIR: cir.store %[[CMP2]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR: %[[A3:.*]] = cir.load %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B3:.*]] = cir.load %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP3:.*]] = cir.cmp(le, %[[A3]], %[[B3]]) : !cir.float, !cir.bool
// CIR: cir.store %[[CMP3]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR: %[[A4:.*]] = cir.load %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B4:.*]] = cir.load %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP4:.*]] = cir.cmp(ge, %[[A4]], %[[B4]]) : !cir.float, !cir.bool
// CIR: cir.store %[[CMP4]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR: %[[A5:.*]] = cir.load %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B5:.*]] = cir.load %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP5:.*]] = cir.cmp(ne, %[[A5]], %[[B5]]) : !cir.float, !cir.bool
// CIR: cir.store %[[CMP5]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// CIR: %[[A6:.*]] = cir.load %[[A_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B6:.*]] = cir.load %[[B_PTR]] : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[CMP6:.*]] = cir.cmp(eq, %[[A6]], %[[B6]]) : !cir.float, !cir.bool
// CIR: cir.store %[[CMP6]], %[[X_PTR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: define void @c0_float(float %0, float %1) {
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
