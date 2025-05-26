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

// CHECK: cir.func @_Z3up0v() -> !u32i
// CHECK:   %[[A:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.unary(plus, %[[INPUT]])

// LLVM: define i32 @_Z3up0v()
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

// CHECK: cir.func @_Z3um0v() -> !u32i
// CHECK:   %[[A:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.unary(minus, %[[INPUT]])

// LLVM: define i32 @_Z3um0v()
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

// CHECK: cir.func @_Z3un0v() -> !u32i
// CHECK:   %[[A:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.unary(not, %[[INPUT]])

// LLVM: define i32 @_Z3un0v()
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

// CHECK: cir.func @_Z4inc0v() -> !s32i
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[INPUT]]) nsw
// CHECK:   cir.store{{.*}} %[[INCREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load{{.*}} %[[A]]

// LLVM: define i32 @_Z4inc0v()
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

// CHECK: cir.func @_Z4dec0v() -> !s32i
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[DECREMENTED:.*]] = cir.unary(dec, %[[INPUT]]) nsw
// CHECK:   cir.store{{.*}} %[[DECREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load{{.*}} %[[A]]

// LLVM: define i32 @_Z4dec0v()
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

// CHECK: cir.func @_Z4inc1v() -> !s32i
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[INPUT]]) nsw
// CHECK:   cir.store{{.*}} %[[INCREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load{{.*}} %[[A]]

// LLVM: define i32 @_Z4inc1v()
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

// CHECK: cir.func @_Z4dec1v() -> !s32i
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[DECREMENTED:.*]] = cir.unary(dec, %[[INPUT]]) nsw
// CHECK:   cir.store{{.*}} %[[DECREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load{{.*}} %[[A]]

// LLVM: define i32 @_Z4dec1v()
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

// CHECK: cir.func @_Z4inc2v() -> !s32i
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[ATOB:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[ATOB]]) nsw
// CHECK:   cir.store{{.*}} %[[INCREMENTED]], %[[A]]
// CHECK:   cir.store{{.*}} %[[ATOB]], %[[B]]
// CHECK:   %[[B_TO_OUTPUT:.*]] = cir.load{{.*}} %[[B]]

// LLVM: define i32 @_Z4inc2v()
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

// CHECK: cir.func @_Z6fpPlusv() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.unary(plus, %[[INPUT]])

// LLVM: define float @_Z6fpPlusv()
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

// CHECK: cir.func @_Z7fpMinusv() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.unary(minus, %[[INPUT]])

// LLVM: define float @_Z7fpMinusv()
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

// CHECK: cir.func @_Z8fpPreIncv() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[INPUT]])

// LLVM: define float @_Z8fpPreIncv()
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

// CHECK: cir.func @_Z8fpPreDecv() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[DECREMENTED:.*]] = cir.unary(dec, %[[INPUT]])

// LLVM: define float @_Z8fpPreDecv()
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

// CHECK: cir.func @_Z9fpPostIncv() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[INPUT]])

// LLVM: define float @_Z9fpPostIncv()
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

// CHECK: cir.func @_Z9fpPostDecv() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[DECREMENTED:.*]] = cir.unary(dec, %[[INPUT]])

// LLVM: define float @_Z9fpPostDecv()
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

// CHECK: cir.func @_Z10fpPostInc2v() -> !cir.float
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[B:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[ATOB:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.unary(inc, %[[ATOB]])
// CHECK:   cir.store{{.*}} %[[INCREMENTED]], %[[A]]
// CHECK:   cir.store{{.*}} %[[ATOB]], %[[B]]
// CHECK:   %[[B_TO_OUTPUT:.*]] = cir.load{{.*}} %[[B]]

// LLVM: define float @_Z10fpPostInc2v()
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

void chars(char c) {
// CHECK: cir.func @_Z5charsc

  int c1 = +c;
  // CHECK: %[[PROMO:.*]] = cir.cast(integral, %{{.+}} : !s8i), !s32i
  // CHECK: cir.unary(plus, %[[PROMO]]) : !s32i, !s32i
  int c2 = -c;
  // CHECK: %[[PROMO:.*]] = cir.cast(integral, %{{.+}} : !s8i), !s32i
  // CHECK: cir.unary(minus, %[[PROMO]]) nsw : !s32i, !s32i

  // Chars can go through some integer promotion codegen paths even when not promoted.
  // These should not have nsw attributes because the intermediate promotion makes the
  // overflow defined behavior.
  ++c; // CHECK: cir.unary(inc, %{{.+}}) : !s8i, !s8i
  --c; // CHECK: cir.unary(dec, %{{.+}}) : !s8i, !s8i
  c++; // CHECK: cir.unary(inc, %{{.+}}) : !s8i, !s8i
  c--; // CHECK: cir.unary(dec, %{{.+}}) : !s8i, !s8i
}

_Float16 fp16UPlus(_Float16 f) {
  return +f;
}

// CHECK: cir.func @_Z9fp16UPlusDF16_({{.*}}) -> !cir.f16
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[F:.*]]
// CHECK:   %[[PROMOTED:.*]] = cir.cast(floating, %[[INPUT]] : !cir.f16), !cir.float
// CHECK:   %[[RESULT:.*]] = cir.unary(plus, %[[PROMOTED]])
// CHECK:   %[[UNPROMOTED:.*]] = cir.cast(floating, %[[RESULT]] : !cir.float), !cir.f16

// LLVM: define half @_Z9fp16UPlusDF16_({{.*}})
// LLVM:   %[[F_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// LLVM:   %[[PROMOTED:.*]] = fpext half %[[F_LOAD]] to float
// LLVM:   %[[UNPROMOTED:.*]] = fptrunc float %[[PROMOTED]] to half

// OGCG: define{{.*}} half @_Z9fp16UPlusDF16_({{.*}})
// OGCG:   %[[F_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// OGCG:   %[[PROMOTED:.*]] = fpext half %[[F_LOAD]] to float
// OGCG:   %[[UNPROMOTED:.*]] = fptrunc float %[[PROMOTED]] to half

_Float16 fp16UMinus(_Float16 f) {
  return -f;
}

// CHECK: cir.func @_Z10fp16UMinusDF16_({{.*}}) -> !cir.f16
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[F:.*]]
// CHECK:   %[[PROMOTED:.*]] = cir.cast(floating, %[[INPUT]] : !cir.f16), !cir.float
// CHECK:   %[[RESULT:.*]] = cir.unary(minus, %[[PROMOTED]])
// CHECK:   %[[UNPROMOTED:.*]] = cir.cast(floating, %[[RESULT]] : !cir.float), !cir.f16

// LLVM: define half @_Z10fp16UMinusDF16_({{.*}})
// LLVM:   %[[F_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// LLVM:   %[[PROMOTED:.*]] = fpext half %[[F_LOAD]] to float
// LLVM:   %[[RESULT:.*]] = fneg float %[[PROMOTED]]
// LLVM:   %[[UNPROMOTED:.*]] = fptrunc float %[[RESULT]] to half

// OGCG: define{{.*}} half @_Z10fp16UMinusDF16_({{.*}})
// OGCG:   %[[F_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// OGCG:   %[[PROMOTED:.*]] = fpext half %[[F_LOAD]] to float
// OGCG:   %[[RESULT:.*]] = fneg float %[[PROMOTED]]
// OGCG:   %[[UNPROMOTED:.*]] = fptrunc float %[[RESULT]] to half

void test_logical_not() {
  int a = 5;
  a = !a;
  bool b = false;
  b = !b;
  float c = 2.0f;
  c = !c;
  int *p = 0;
  b = !p;
  double d = 3.0;
  b = !d;
}

// CHECK: cir.func @_Z16test_logical_notv()
// CHECK:   %[[A:.*]] = cir.load{{.*}} %[[A_ADDR:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[A_BOOL:.*]] = cir.cast(int_to_bool, %[[A]] : !s32i), !cir.bool
// CHECK:   %[[A_NOT:.*]] = cir.unary(not, %[[A_BOOL]]) : !cir.bool, !cir.bool
// CHECK:   %[[A_CAST:.*]] = cir.cast(bool_to_int, %[[A_NOT]] : !cir.bool), !s32i
// CHECK:   cir.store{{.*}} %[[A_CAST]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[B:.*]] = cir.load{{.*}} %[[B_ADDR:.*]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:   %[[B_NOT:.*]] = cir.unary(not, %[[B]]) : !cir.bool, !cir.bool
// CHECK:   cir.store{{.*}} %[[B_NOT]], %[[B_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:   %[[C:.*]] = cir.load{{.*}} %[[C_ADDR:.*]] : !cir.ptr<!cir.float>, !cir.float
// CHECK:   %[[C_BOOL:.*]] = cir.cast(float_to_bool, %[[C]] : !cir.float), !cir.bool
// CHECK:   %[[C_NOT:.*]] = cir.unary(not, %[[C_BOOL]]) : !cir.bool, !cir.bool
// CHECK:   %[[C_CAST:.*]] = cir.cast(bool_to_float, %[[C_NOT]] : !cir.bool), !cir.float
// CHECK:   cir.store{{.*}} %[[C_CAST]], %[[C_ADDR]] : !cir.float, !cir.ptr<!cir.float>
// CHECK:   %[[P:.*]] = cir.load{{.*}} %[[P_ADDR:.*]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:   %[[P_BOOL:.*]] = cir.cast(ptr_to_bool, %[[P]] : !cir.ptr<!s32i>), !cir.bool
// CHECK:   %[[P_NOT:.*]] = cir.unary(not, %[[P_BOOL]]) : !cir.bool, !cir.bool
// CHECK:   cir.store{{.*}} %[[P_NOT]], %[[B_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:   %[[D:.*]] = cir.load{{.*}} %[[D_ADDR:.*]] : !cir.ptr<!cir.double>, !cir.double
// CHECK:   %[[D_BOOL:.*]] = cir.cast(float_to_bool, %[[D]] : !cir.double), !cir.bool
// CHECK:   %[[D_NOT:.*]] = cir.unary(not, %[[D_BOOL]]) : !cir.bool, !cir.bool
// CHECK:   cir.store{{.*}} %[[D_NOT]], %[[B_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: define void @_Z16test_logical_notv()
// LLVM:   %[[A:.*]] = load i32, ptr %[[A_ADDR:.*]], align 4
// LLVM:   %[[A_BOOL:.*]] = icmp ne i32 %[[A]], 0
// LLVM:   %[[A_NOT:.*]] = xor i1 %[[A_BOOL]], true
// LLVM:   %[[A_CAST:.*]] = zext i1 %[[A_NOT]] to i32
// LLVM:   store i32 %[[A_CAST]], ptr %[[A_ADDR]], align 4
// LLVM:   %[[B:.*]] = load i8, ptr %[[B_ADDR:.*]], align 1
// LLVM:   %[[B_BOOL:.*]] = trunc i8 %[[B]] to i1
// LLVM:   %[[B_NOT:.*]] = xor i1 %[[B_BOOL]], true
// LLVM:   %[[B_CAST:.*]] = zext i1 %[[B_NOT]] to i8
// LLVM:   store i8 %[[B_CAST]], ptr %[[B_ADDR]], align 1
// LLVM:   %[[C:.*]] = load float, ptr %[[C_ADDR:.*]], align 4
// LLVM:   %[[C_BOOL:.*]] = fcmp une float %[[C]], 0.000000e+00
// LLVM:   %[[C_NOT:.*]] = xor i1 %[[C_BOOL]], true
// LLVM:   %[[C_CAST:.*]] = uitofp i1 %[[C_NOT]] to float
// LLVM:   store float %[[C_CAST]], ptr %[[C_ADDR]], align 4
// LLVM:   %[[P:.*]] = load ptr, ptr %[[P_ADDR:.*]], align 8
// LLVM:   %[[P_BOOL:.*]] = icmp ne ptr %[[P]], null
// LLVM:   %[[P_NOT:.*]] = xor i1 %[[P_BOOL]], true
// LLVM:   %[[P_CAST:.*]] = zext i1 %[[P_NOT]] to i8
// LLVM:   store i8 %[[P_CAST]], ptr %[[B_ADDR]], align 1
// LLVM:   %[[D:.*]] = load double, ptr %[[D_ADDR:.*]], align 8
// LLVM:   %[[D_BOOL:.*]] = fcmp une double %[[D]], 0.000000e+00
// LLVM:   %[[D_NOT:.*]] = xor i1 %[[D_BOOL]], true
// LLVM:   %[[D_CAST:.*]] = zext i1 %[[D_NOT]] to i8
// LLVM:   store i8 %[[D_CAST]], ptr %[[B_ADDR]], align 1

// OGCG: define{{.*}} void @_Z16test_logical_notv()
// OGCG:   %[[A:.*]] = load i32, ptr %[[A_ADDR:.*]], align 4
// OGCG:   %[[A_BOOL:.*]] = icmp ne i32 %[[A]], 0
// OGCG:   %[[A_NOT:.*]] = xor i1 %[[A_BOOL]], true
// OGCG:   %[[A_CAST:.*]] = zext i1 %[[A_NOT]] to i32
// OGCG:   store i32 %[[A_CAST]], ptr %[[A_ADDR]], align 4
// OGCG:   %[[B:.*]] = load i8, ptr %[[B_ADDR:.*]], align 1
// OGCG:   %[[B_BOOL:.*]] = trunc i8 %[[B]] to i1
// OGCG:   %[[B_NOT:.*]] = xor i1 %[[B_BOOL]], true
// OGCG:   %[[B_CAST:.*]] = zext i1 %[[B_NOT]] to i8
// OGCG:   store i8 %[[B_CAST]], ptr %[[B_ADDR]], align 1
// OGCG:   %[[C:.*]] = load float, ptr %[[C_ADDR:.*]], align 4
// OGCG:   %[[C_BOOL:.*]] = fcmp une float %[[C]], 0.000000e+00
// OGCG:   %[[C_NOT:.*]] = xor i1 %[[C_BOOL]], true
// OGCG:   %[[C_CAST:.*]] = uitofp i1 %[[C_NOT]] to float
// OGCG:   store float %[[C_CAST]], ptr %[[C_ADDR]], align 4
// OGCG:   %[[P:.*]] = load ptr, ptr %[[P_ADDR:.*]], align 8
// OGCG:   %[[P_BOOL:.*]] = icmp ne ptr %[[P]], null
// OGCG:   %[[P_NOT:.*]] = xor i1 %[[P_BOOL]], true
// OGCG:   %[[P_CAST:.*]] = zext i1 %[[P_NOT]] to i8
// OGCG:   store i8 %[[P_CAST]], ptr %[[B_ADDR]], align 1
// OGCG:   %[[D:.*]] = load double, ptr %[[D_ADDR:.*]], align 8
// OGCG:   %[[D_BOOL:.*]] = fcmp une double %[[D]], 0.000000e+00
// OGCG:   %[[D_NOT:.*]] = xor i1 %[[D_BOOL]], true
// OGCG:   %[[D_CAST:.*]] = zext i1 %[[D_NOT]] to i8
// OGCG:   store i8 %[[D_CAST]], ptr %[[B_ADDR]], align 1
