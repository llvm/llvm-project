// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -Wno-unused-value -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

unsigned up0() {
  unsigned a = 1u;
  return +a;
}

// CHECK: cir.func @up0() -> !u32i
// CHECK:   %[[A:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.unary(plus, %[[INPUT]])

// LLVM: define i32 @up0()
// LLVM:   %[[RV:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4

// OGCG: define{{.*}} i32 @_Z3up0v()
// OGCG:   %[[A:.*]] = alloca i32, align 4
// OGCG:   store i32 1, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4

unsigned um0() {
  unsigned a = 1u;
  return -a;
}

// CHECK: cir.func @um0() -> !u32i
// CHECK:   %[[A:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.unary(minus, %[[INPUT]])

// LLVM: define i32 @um0()
// LLVM:   %[[RV:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// LLVM:   %[[RESULT:.*]] = sub i32 0, %[[A_LOAD]]

// OGCG: define{{.*}} i32 @_Z3um0v()
// OGCG:   %[[A:.*]] = alloca i32, align 4
// OGCG:   store i32 1, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// OGCG:   %[[RESULT:.*]] = sub i32 0, %[[A_LOAD]]

unsigned un0() {
  unsigned a = 1u;
  return ~a; // a ^ -1 , not
}

// CHECK: cir.func @un0() -> !u32i
// CHECK:   %[[A:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.unary(not, %[[INPUT]])

// LLVM: define i32 @un0()
// LLVM:   %[[RV:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// LLVM:   %[[RESULT:.*]] = xor i32 %[[A_LOAD]], -1

// OGCG: define{{.*}} i32 @_Z3un0v()
// OGCG:   %[[A:.*]] = alloca i32, align 4
// OGCG:   store i32 1, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// OGCG:   %[[RESULT:.*]] = xor i32 %[[A_LOAD]], -1

int inc0() {
  int a = 1;
  ++a;
  return a;
}

// CHECK: cir.func @inc0() -> !s32i
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[INPUT]])
// CHECK:   cir.store %[[INCREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load %[[A]]

// LLVM: define i32 @inc0()
// LLVM:   %[[RV:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// LLVM:   %[[RESULT:.*]] = add nsw i32 %[[A_LOAD]], 1

// OGCG: define{{.*}} i32 @_Z4inc0v()
// OGCG:   %[[A:.*]] = alloca i32, align 4
// OGCG:   store i32 1, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// OGCG:   %[[RESULT:.*]] = add nsw i32 %[[A_LOAD]], 1

int dec0() {
  int a = 1;
  --a;
  return a;
}

// CHECK: cir.func @dec0() -> !s32i
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(dec, %[[INPUT]])
// CHECK:   cir.store %[[INCREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load %[[A]]

// LLVM: define i32 @dec0()
// LLVM:   %[[RV:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// LLVM:   %[[RESULT:.*]] = sub nsw i32 %[[A_LOAD]], 1

// OGCG: define{{.*}} i32 @_Z4dec0v()
// OGCG:   %[[A:.*]] = alloca i32, align 4
// OGCG:   store i32 1, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// OGCG:   %[[RESULT:.*]] = add nsw i32 %[[A_LOAD]], -1

int inc1() {
  int a = 1;
  a++;
  return a;
}

// CHECK: cir.func @inc1() -> !s32i
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[INPUT]])
// CHECK:   cir.store %[[INCREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load %[[A]]

// LLVM: define i32 @inc1()
// LLVM:   %[[RV:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// LLVM:   %[[RESULT:.*]] = add nsw i32 %[[A_LOAD]], 1

// OGCG: define{{.*}} i32 @_Z4inc1v()
// OGCG:   %[[A:.*]] = alloca i32, align 4
// OGCG:   store i32 1, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// OGCG:   %[[RESULT:.*]] = add nsw i32 %[[A_LOAD]], 1

int dec1() {
  int a = 1;
  a--;
  return a;
}

// CHECK: cir.func @dec1() -> !s32i
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(dec, %[[INPUT]])
// CHECK:   cir.store %[[INCREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load %[[A]]

// LLVM: define i32 @dec1()
// LLVM:   %[[RV:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// LLVM:   %[[RESULT:.*]] = sub nsw i32 %[[A_LOAD]], 1

// OGCG: define{{.*}} i32 @_Z4dec1v()
// OGCG:   %[[A:.*]] = alloca i32, align 4
// OGCG:   store i32 1, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// OGCG:   %[[RESULT:.*]] = add nsw i32 %[[A_LOAD]], -1

// Ensure the increment is performed after the assignment to b.
int inc2() {
  int a = 1;
  int b = a++;
  return b;
}

// CHECK: cir.func @inc2() -> !s32i
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[ATOB:.*]] = cir.load %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[ATOB]])
// CHECK:   cir.store %[[INCREMENTED]], %[[A]]
// CHECK:   cir.store %[[ATOB]], %[[B]]
// CHECK:   %[[B_TO_OUTPUT:.*]] = cir.load %[[B]]

// LLVM: define i32 @inc2()
// LLVM:   %[[RV:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[B:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// LLVM:   %[[A_INC:.*]] = add nsw i32 %[[A_LOAD]], 1
// LLVM:   store i32 %[[A_INC]], ptr %[[A]], align 4
// LLVM:   store i32 %[[A_LOAD]], ptr %[[B]], align 4
// LLVM:   %[[B_TO_OUTPUT:.*]] = load i32, ptr %[[B]], align 4

// OGCG: define{{.*}} i32 @_Z4inc2v()
// OGCG:   %[[A:.*]] = alloca i32, align 4
// OGCG:   %[[B:.*]] = alloca i32, align 4
// OGCG:   store i32 1, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load i32, ptr %[[A]], align 4
// OGCG:   %[[A_INC:.*]] = add nsw i32 %[[A_LOAD]], 1
// OGCG:   store i32 %[[A_INC]], ptr %[[A]], align 4
// OGCG:   store i32 %[[A_LOAD]], ptr %[[B]], align 4
// OGCG:   %[[B_TO_OUTPUT:.*]] = load i32, ptr %[[B]], align 4

float fpPlus() {
  float a = 1.0f;
  return +a;
}

// CHECK: cir.func @fpPlus() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.unary(plus, %[[INPUT]])

// LLVM: define float @fpPlus()
// LLVM:   %[[RV:.*]] = alloca float, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca float, i64 1, align 4
// LLVM:   store float 1.000000e+00, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4

// OGCG: define{{.*}} float @_Z6fpPlusv()
// OGCG:   %[[A:.*]] = alloca float, align 4
// OGCG:   store float 1.000000e+00, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4

float fpMinus() {
  float a = 1.0f;
  return -a;
}

// CHECK: cir.func @fpMinus() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.unary(minus, %[[INPUT]])

// LLVM: define float @fpMinus()
// LLVM:   %[[RV:.*]] = alloca float, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca float, i64 1, align 4
// LLVM:   store float 1.000000e+00, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// LLVM:   %[[RESULT:.*]] = fneg float %[[A_LOAD]]

// OGCG: define{{.*}} float @_Z7fpMinusv()
// OGCG:   %[[A:.*]] = alloca float, align 4
// OGCG:   store float 1.000000e+00, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// OGCG:   %[[RESULT:.*]] = fneg float %[[A_LOAD]]

float fpPreInc() {
  float a = 1.0f;
  return ++a;
}

// CHECK: cir.func @fpPreInc() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[INPUT]])

// LLVM: define float @fpPreInc()
// LLVM:   %[[RV:.*]] = alloca float, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca float, i64 1, align 4
// LLVM:   store float 1.000000e+00, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// LLVM:   %[[RESULT:.*]] = fadd float 1.000000e+00, %[[A_LOAD]]

// OGCG: define{{.*}} float @_Z8fpPreIncv()
// OGCG:   %[[A:.*]] = alloca float, align 4
// OGCG:   store float 1.000000e+00, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// OGCG:   %[[RESULT:.*]] = fadd float %[[A_LOAD]], 1.000000e+00

float fpPreDec() {
  float a = 1.0f;
  return --a;
}

// CHECK: cir.func @fpPreDec() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[DECREMENTED:.*]] = cir.unary(dec, %[[INPUT]])

// LLVM: define float @fpPreDec()
// LLVM:   %[[RV:.*]] = alloca float, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca float, i64 1, align 4
// LLVM:   store float 1.000000e+00, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// LLVM:   %[[RESULT:.*]] = fadd float -1.000000e+00, %[[A_LOAD]]

// OGCG: define{{.*}} float @_Z8fpPreDecv()
// OGCG:   %[[A:.*]] = alloca float, align 4
// OGCG:   store float 1.000000e+00, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// OGCG:   %[[RESULT:.*]] = fadd float %[[A_LOAD]], -1.000000e+00

float fpPostInc() {
  float a = 1.0f;
  return a++;
}

// CHECK: cir.func @fpPostInc() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[INPUT]])

// LLVM: define float @fpPostInc()
// LLVM:   %[[RV:.*]] = alloca float, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca float, i64 1, align 4
// LLVM:   store float 1.000000e+00, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// LLVM:   %[[RESULT:.*]] = fadd float 1.000000e+00, %[[A_LOAD]]

// OGCG: define{{.*}} float @_Z9fpPostIncv()
// OGCG:   %[[A:.*]] = alloca float, align 4
// OGCG:   store float 1.000000e+00, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// OGCG:   %[[RESULT:.*]] = fadd float %[[A_LOAD]], 1.000000e+00

float fpPostDec() {
  float a = 1.0f;
  return a--;
}

// CHECK: cir.func @fpPostDec() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load %[[A]]
// CHECK:   %[[DECREMENTED:.*]] = cir.unary(dec, %[[INPUT]])

// LLVM: define float @fpPostDec()
// LLVM:   %[[RV:.*]] = alloca float, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca float, i64 1, align 4
// LLVM:   store float 1.000000e+00, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// LLVM:   %[[RESULT:.*]] = fadd float -1.000000e+00, %[[A_LOAD]]

// OGCG: define{{.*}} float @_Z9fpPostDecv()
// OGCG:   %[[A:.*]] = alloca float, align 4
// OGCG:   store float 1.000000e+00, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// OGCG:   %[[RESULT:.*]] = fadd float %[[A_LOAD]], -1.000000e+00

// Ensure the increment is performed after the assignment to b.
float fpPostInc2() {
  float a = 1.0f;
  float b = a++;
  return b;
}

// CHECK: cir.func @fpPostInc2() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[B:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[ATOB:.*]] = cir.load %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[ATOB]])
// CHECK:   cir.store %[[INCREMENTED]], %[[A]]
// CHECK:   cir.store %[[ATOB]], %[[B]]
// CHECK:   %[[B_TO_OUTPUT:.*]] = cir.load %[[B]]

// LLVM: define float @fpPostInc2()
// LLVM:   %[[RV:.*]] = alloca float, i64 1, align 4
// LLVM:   %[[A:.*]] = alloca float, i64 1, align 4
// LLVM:   %[[B:.*]] = alloca float, i64 1, align 4
// LLVM:   store float 1.000000e+00, ptr %[[A]], align 4
// LLVM:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// LLVM:   %[[A_INC:.*]] = fadd float 1.000000e+00, %[[A_LOAD]]
// LLVM:   store float %[[A_INC]], ptr %[[A]], align 4
// LLVM:   store float %[[A_LOAD]], ptr %[[B]], align 4
// LLVM:   %[[B_TO_OUTPUT:.*]] = load float, ptr %[[B]], align 4

// OGCG: define{{.*}} float @_Z10fpPostInc2v()
// OGCG:   %[[A:.*]] = alloca float, align 4
// OGCG:   %[[B:.*]] = alloca float, align 4
// OGCG:   store float 1.000000e+00, ptr %[[A]], align 4
// OGCG:   %[[A_LOAD:.*]] = load float, ptr %[[A]], align 4
// OGCG:   %[[A_INC:.*]] = fadd float %[[A_LOAD]], 1.000000e+00
// OGCG:   store float %[[A_INC]], ptr %[[A]], align 4
// OGCG:   store float %[[A_LOAD]], ptr %[[B]], align 4
// OGCG:   %[[B_TO_OUTPUT:.*]] = load float, ptr %[[B]], align 4
