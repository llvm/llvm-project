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

// CHECK: cir.func{{.*}} @_Z3up0v() -> (!u32i{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]

// LLVM: define{{.*}} i32 @_Z3up0v()
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

// CHECK: cir.func{{.*}} @_Z3um0v() -> (!u32i{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.minus %[[INPUT]]

// LLVM: define{{.*}} i32 @_Z3um0v()
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

// CHECK: cir.func{{.*}} @_Z3un0v() -> (!u32i{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.not %[[INPUT]]

// LLVM: define{{.*}} i32 @_Z3un0v()
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

// CHECK: cir.func{{.*}} @_Z4inc0v() -> (!s32i{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.inc nsw %[[INPUT]]
// CHECK:   cir.store{{.*}} %[[INCREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load{{.*}} %[[A]]

// LLVM: define{{.*}} i32 @_Z4inc0v()
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

// CHECK: cir.func{{.*}} @_Z4dec0v() -> (!s32i{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[DECREMENTED:.*]] = cir.dec nsw %[[INPUT]]
// CHECK:   cir.store{{.*}} %[[DECREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load{{.*}} %[[A]]

// LLVM: define{{.*}} i32 @_Z4dec0v()
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

// CHECK: cir.func{{.*}} @_Z4inc1v() -> (!s32i{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.inc nsw %[[INPUT]]
// CHECK:   cir.store{{.*}} %[[INCREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load{{.*}} %[[A]]

// LLVM: define{{.*}} i32 @_Z4inc1v()
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

// CHECK: cir.func{{.*}} @_Z4dec1v() -> (!s32i{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[DECREMENTED:.*]] = cir.dec nsw %[[INPUT]]
// CHECK:   cir.store{{.*}} %[[DECREMENTED]], %[[A]]
// CHECK:   %[[A_TO_OUTPUT:.*]] = cir.load{{.*}} %[[A]]

// LLVM: define{{.*}} i32 @_Z4dec1v()
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

// CHECK: cir.func{{.*}} @_Z4inc2v() -> (!s32i{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[B:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !s32i
// CHECK:   %[[ATOB:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.inc nsw %[[ATOB]]
// CHECK:   cir.store{{.*}} %[[INCREMENTED]], %[[A]]
// CHECK:   cir.store{{.*}} %[[ATOB]], %[[B]]
// CHECK:   %[[B_TO_OUTPUT:.*]] = cir.load{{.*}} %[[B]]

// LLVM: define{{.*}} i32 @_Z4inc2v()
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

void chars(char c) {
// CHECK: cir.func{{.*}} @_Z5charsc

  int c1 = +c;
  // CHECK: %[[PROMO:.*]] = cir.cast integral %{{.+}} : !s8i -> !s32i
  int c2 = -c;
  // CHECK: %[[PROMO:.*]] = cir.cast integral %{{.+}} : !s8i -> !s32i
  // CHECK: cir.minus nsw %[[PROMO]] : !s32i

  // Chars can go through some integer promotion codegen paths even when not promoted.
  // These should not have nsw attributes because the intermediate promotion makes the
  // overflow defined behavior.
  ++c; // CHECK: cir.inc %{{.+}} : !s8i
  --c; // CHECK: cir.dec %{{.+}} : !s8i
  c++; // CHECK: cir.inc %{{.+}} : !s8i
  c--; // CHECK: cir.dec %{{.+}} : !s8i
}

float fpPlus() {
  float a = 1.0f;
  return +a;
}

// CHECK: cir.func{{.*}} @_Z6fpPlusv() -> (!cir.float{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]

// LLVM: define{{.*}} float @_Z6fpPlusv()
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

// CHECK: cir.func{{.*}} @_Z7fpMinusv() -> (!cir.float{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[OUTPUT:.*]] = cir.minus %[[INPUT]]

// LLVM: define{{.*}} float @_Z7fpMinusv()
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

// CHECK: cir.func{{.*}} @_Z8fpPreIncv() -> (!cir.float{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.inc %[[INPUT]]

// LLVM: define{{.*}} float @_Z8fpPreIncv()
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

// CHECK: cir.func{{.*}} @_Z8fpPreDecv() -> (!cir.float{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[DECREMENTED:.*]] = cir.dec %[[INPUT]]

// LLVM: define{{.*}} float @_Z8fpPreDecv()
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

// CHECK: cir.func{{.*}} @_Z9fpPostIncv() -> (!cir.float{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.inc %[[INPUT]]

// LLVM: define{{.*}} float @_Z9fpPostIncv()
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

// CHECK: cir.func{{.*}} @_Z9fpPostDecv() -> (!cir.float{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[DECREMENTED:.*]] = cir.dec %[[INPUT]]

// LLVM: define{{.*}} float @_Z9fpPostDecv()
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

// CHECK: cir.func{{.*}} @_Z10fpPostInc2v() -> (!cir.float{{.*}})
// CHECK:   %[[A:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["a", init]
// CHECK:   %[[B:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["b", init]
// CHECK:   %[[ATMP:.*]] = cir.const #cir.fp<1.000000e+00> : !cir.float
// CHECK:   cir.store{{.*}} %[[ATMP]], %[[A]] : !cir.float
// CHECK:   %[[ATOB:.*]] = cir.load{{.*}} %[[A]]
// CHECK:   %[[INCREMENTED:.*]] = cir.inc %[[ATOB]]
// CHECK:   cir.store{{.*}} %[[INCREMENTED]], %[[A]]
// CHECK:   cir.store{{.*}} %[[ATOB]], %[[B]]
// CHECK:   %[[B_TO_OUTPUT:.*]] = cir.load{{.*}} %[[B]]

// LLVM: define{{.*}} float @_Z10fpPostInc2v()
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

// double unary operations
double doubleUPlus(double f) {
  return +f;
}

// CHECK: cir.func{{.*}} @_Z11doubleUPlusd({{.*}}) -> (!cir.double{{.*}})
// CHECK:   %[[DBL_F:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["f", init]
// CHECK:   %[[DBL_LOAD:.*]] = cir.load{{.*}} %[[DBL_F]]

// LLVM: define{{.*}} double @_Z11doubleUPlusd({{.*}})
// LLVM:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8

// OGCG: define{{.*}} double @_Z11doubleUPlusd({{.*}})
// OGCG:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8

double doubleUMinus(double f) {
  return -f;
}

// CHECK: cir.func{{.*}} @_Z12doubleUMinusd({{.*}}) -> (!cir.double{{.*}})
// CHECK:   %[[DBL_F:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["f", init]
// CHECK:   %[[DBL_LOAD:.*]] = cir.load{{.*}} %[[DBL_F]]
// CHECK:   %[[DBL_NEGATED:.*]] = cir.minus %[[DBL_LOAD]]

// LLVM: define{{.*}} double @_Z12doubleUMinusd({{.*}})
// LLVM:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8
// LLVM:   %[[DBL_NEGATED:.*]] = fneg double %[[DBL_LOAD]]

// OGCG: define{{.*}} double @_Z12doubleUMinusd({{.*}})
// OGCG:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8
// OGCG:   %[[DBL_NEGATED:.*]] = fneg double %[[DBL_LOAD]]

double doubleUPreInc(double f) {
  return ++f;
}

// CHECK: cir.func{{.*}} @_Z13doubleUPreIncd({{.*}}) -> (!cir.double{{.*}})
// CHECK:   %[[DBL_F:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["f", init]
// CHECK:   %[[DBL_LOAD:.*]] = cir.load{{.*}} %[[DBL_F]]
// CHECK:   %[[DBL_INC:.*]] = cir.inc %[[DBL_LOAD]] : !cir.double

// LLVM: define{{.*}} double @_Z13doubleUPreIncd({{.*}})
// LLVM:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8
// LLVM:   %[[DBL_INC:.*]] = fadd double 1.000000e+00, %[[DBL_LOAD]]

// OGCG: define{{.*}} double @_Z13doubleUPreIncd({{.*}})
// OGCG:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8
// OGCG:   %[[DBL_INC:.*]] = fadd double %[[DBL_LOAD]], 1.000000e+00

double doubleUPreDec(double f) {
  return --f;
}

// CHECK: cir.func{{.*}} @_Z13doubleUPreDecd({{.*}}) -> (!cir.double{{.*}})
// CHECK:   %[[DBL_F:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["f", init]
// CHECK:   %[[DBL_LOAD:.*]] = cir.load{{.*}} %[[DBL_F]]
// CHECK:   %[[DBL_DEC:.*]] = cir.dec %[[DBL_LOAD]] : !cir.double

// LLVM: define{{.*}} double @_Z13doubleUPreDecd({{.*}})
// LLVM:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8
// LLVM:   %[[DBL_DEC:.*]] = fadd double -1.000000e+00, %[[DBL_LOAD]]

// OGCG: define{{.*}} double @_Z13doubleUPreDecd({{.*}})
// OGCG:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8
// OGCG:   %[[DBL_DEC:.*]] = fadd double %[[DBL_LOAD]], -1.000000e+00

double doubleUPostInc(double f) {
  return f++;
}

// CHECK: cir.func{{.*}} @_Z14doubleUPostIncd({{.*}}) -> (!cir.double{{.*}})
// CHECK:   %[[DBL_F:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["f", init]
// CHECK:   %[[DBL_LOAD:.*]] = cir.load{{.*}} %[[DBL_F]]
// CHECK:   %[[DBL_INC:.*]] = cir.inc %[[DBL_LOAD]] : !cir.double

// LLVM: define{{.*}} double @_Z14doubleUPostIncd({{.*}})
// LLVM:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8
// LLVM:   %[[DBL_INC:.*]] = fadd double 1.000000e+00, %[[DBL_LOAD]]

// OGCG: define{{.*}} double @_Z14doubleUPostIncd({{.*}})
// OGCG:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8
// OGCG:   %[[DBL_INC:.*]] = fadd double %[[DBL_LOAD]], 1.000000e+00

double doubleUPostDec(double f) {
  return f--;
}

// CHECK: cir.func{{.*}} @_Z14doubleUPostDecd({{.*}}) -> (!cir.double{{.*}})
// CHECK:   %[[DBL_F:.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["f", init]
// CHECK:   %[[DBL_LOAD:.*]] = cir.load{{.*}} %[[DBL_F]]
// CHECK:   %[[DBL_DEC:.*]] = cir.dec %[[DBL_LOAD]] : !cir.double

// LLVM: define{{.*}} double @_Z14doubleUPostDecd({{.*}})
// LLVM:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8
// LLVM:   %[[DBL_DEC:.*]] = fadd double -1.000000e+00, %[[DBL_LOAD]]

// OGCG: define{{.*}} double @_Z14doubleUPostDecd({{.*}})
// OGCG:   %[[DBL_LOAD:.*]] = load double, ptr %{{.*}}, align 8
// OGCG:   %[[DBL_DEC:.*]] = fadd double %[[DBL_LOAD]], -1.000000e+00

// long double unary operations
long double ldUPlus(long double f) {
  return +f;
}

// CHECK: cir.func{{.*}} @_Z7ldUPluse({{.*}}) -> (!cir.long_double<!cir.f80>{{.*}})
// CHECK:   %[[LD_F:.*]] = cir.alloca !cir.long_double<!cir.f80>, !cir.ptr<!cir.long_double<!cir.f80>>, ["f", init]
// CHECK:   %[[LD_LOAD:.*]] = cir.load{{.*}} %[[LD_F]]

// LLVM: define{{.*}} x86_fp80 @_Z7ldUPluse({{.*}})
// LLVM:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16

// OGCG: define{{.*}} x86_fp80 @_Z7ldUPluse({{.*}})
// OGCG:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16

long double ldUMinus(long double f) {
  return -f;
}

// CHECK: cir.func{{.*}} @_Z8ldUMinuse({{.*}}) -> (!cir.long_double<!cir.f80>{{.*}})
// CHECK:   %[[LD_F:.*]] = cir.alloca !cir.long_double<!cir.f80>, !cir.ptr<!cir.long_double<!cir.f80>>, ["f", init]
// CHECK:   %[[LD_LOAD:.*]] = cir.load{{.*}} %[[LD_F]]
// CHECK:   %[[LD_NEGATED:.*]] = cir.minus %[[LD_LOAD]]

// LLVM: define{{.*}} x86_fp80 @_Z8ldUMinuse({{.*}})
// LLVM:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16
// LLVM:   %[[LD_NEGATED:.*]] = fneg x86_fp80 %[[LD_LOAD]]

// OGCG: define{{.*}} x86_fp80 @_Z8ldUMinuse({{.*}})
// OGCG:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16
// OGCG:   %[[LD_NEGATED:.*]] = fneg x86_fp80 %[[LD_LOAD]]

long double ldUPreInc(long double f) {
  return ++f;
}

// CHECK: cir.func{{.*}} @_Z9ldUPreInce({{.*}}) -> (!cir.long_double<!cir.f80>{{.*}})
// CHECK:   %[[LD_F:.*]] = cir.alloca !cir.long_double<!cir.f80>, !cir.ptr<!cir.long_double<!cir.f80>>, ["f", init]
// CHECK:   %[[LD_LOAD:.*]] = cir.load{{.*}} %[[LD_F]]
// CHECK:   %[[LD_INC:.*]] = cir.inc %[[LD_LOAD]] : !cir.long_double<!cir.f80>

// LLVM: define{{.*}} x86_fp80 @_Z9ldUPreInce({{.*}})
// LLVM:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16
// LLVM:   %[[LD_INC:.*]] = fadd x86_fp80 0xK3FFF8000000000000000, %[[LD_LOAD]]

// OGCG: define{{.*}} x86_fp80 @_Z9ldUPreInce({{.*}})
// OGCG:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16
// OGCG:   %[[LD_INC:.*]] = fadd x86_fp80 %[[LD_LOAD]], 0xK3FFF8000000000000000

long double ldUPreDec(long double f) {
  return --f;
}

// CHECK: cir.func{{.*}} @_Z9ldUPreDece({{.*}}) -> (!cir.long_double<!cir.f80>{{.*}})
// CHECK:   %[[LD_F:.*]] = cir.alloca !cir.long_double<!cir.f80>, !cir.ptr<!cir.long_double<!cir.f80>>, ["f", init]
// CHECK:   %[[LD_LOAD:.*]] = cir.load{{.*}} %[[LD_F]]
// CHECK:   %[[LD_DEC:.*]] = cir.dec %[[LD_LOAD]] : !cir.long_double<!cir.f80>

// LLVM: define{{.*}} x86_fp80 @_Z9ldUPreDece({{.*}})
// LLVM:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16
// LLVM:   %[[LD_DEC:.*]] = fadd x86_fp80 0xKBFFF8000000000000000, %[[LD_LOAD]]

// OGCG: define{{.*}} x86_fp80 @_Z9ldUPreDece({{.*}})
// OGCG:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16
// OGCG:   %[[LD_DEC:.*]] = fadd x86_fp80 %[[LD_LOAD]], 0xKBFFF8000000000000000

long double ldUPostInc(long double f) {
  return f++;
}

// CHECK: cir.func{{.*}} @_Z10ldUPostInce({{.*}}) -> (!cir.long_double<!cir.f80>{{.*}})
// CHECK:   %[[LD_F:.*]] = cir.alloca !cir.long_double<!cir.f80>, !cir.ptr<!cir.long_double<!cir.f80>>, ["f", init]
// CHECK:   %[[LD_LOAD:.*]] = cir.load{{.*}} %[[LD_F]]
// CHECK:   %[[LD_INC:.*]] = cir.inc %[[LD_LOAD]] : !cir.long_double<!cir.f80>

// LLVM: define{{.*}} x86_fp80 @_Z10ldUPostInce({{.*}})
// LLVM:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16
// LLVM:   %[[LD_INC:.*]] = fadd x86_fp80 0xK3FFF8000000000000000, %[[LD_LOAD]]

// OGCG: define{{.*}} x86_fp80 @_Z10ldUPostInce({{.*}})
// OGCG:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16
// OGCG:   %[[LD_INC:.*]] = fadd x86_fp80 %[[LD_LOAD]], 0xK3FFF8000000000000000

long double ldUPostDec(long double f) {
  return f--;
}

// CHECK: cir.func{{.*}} @_Z10ldUPostDece({{.*}}) -> (!cir.long_double<!cir.f80>{{.*}})
// CHECK:   %[[LD_F:.*]] = cir.alloca !cir.long_double<!cir.f80>, !cir.ptr<!cir.long_double<!cir.f80>>, ["f", init]
// CHECK:   %[[LD_LOAD:.*]] = cir.load{{.*}} %[[LD_F]]
// CHECK:   %[[LD_DEC:.*]] = cir.dec %[[LD_LOAD]] : !cir.long_double<!cir.f80>

// LLVM: define{{.*}} x86_fp80 @_Z10ldUPostDece({{.*}})
// LLVM:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16
// LLVM:   %[[LD_DEC:.*]] = fadd x86_fp80 0xKBFFF8000000000000000, %[[LD_LOAD]]

// OGCG: define{{.*}} x86_fp80 @_Z10ldUPostDece({{.*}})
// OGCG:   %[[LD_LOAD:.*]] = load x86_fp80, ptr %{{.*}}, align 16
// OGCG:   %[[LD_DEC:.*]] = fadd x86_fp80 %[[LD_LOAD]], 0xKBFFF8000000000000000

// __float128 unary operations
__float128 f128UPlus(__float128 f) {
  return +f;
}

// CHECK: cir.func{{.*}} @_Z9f128UPlusg({{.*}}) -> (!cir.f128{{.*}})
// CHECK:   %[[F128_F:.*]] = cir.alloca !cir.f128, !cir.ptr<!cir.f128>, ["f", init]
// CHECK:   %[[F128_LOAD:.*]] = cir.load{{.*}} %[[F128_F]]

// LLVM: define{{.*}} fp128 @_Z9f128UPlusg({{.*}})
// LLVM:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16

// OGCG: define{{.*}} fp128 @_Z9f128UPlusg({{.*}})
// OGCG:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16

__float128 f128UMinus(__float128 f) {
  return -f;
}

// CHECK: cir.func{{.*}} @_Z10f128UMinusg({{.*}}) -> (!cir.f128{{.*}})
// CHECK:   %[[F128_F:.*]] = cir.alloca !cir.f128, !cir.ptr<!cir.f128>, ["f", init]
// CHECK:   %[[F128_LOAD:.*]] = cir.load{{.*}} %[[F128_F]]
// CHECK:   %[[F128_NEG:.*]] = cir.minus %[[F128_LOAD]]

// LLVM: define{{.*}} fp128 @_Z10f128UMinusg({{.*}})
// LLVM:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16
// LLVM:   %[[F128_NEG:.*]] = fneg fp128 %[[F128_LOAD]]

// OGCG: define{{.*}} fp128 @_Z10f128UMinusg({{.*}})
// OGCG:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16
// OGCG:   %[[F128_NEG:.*]] = fneg fp128 %[[F128_LOAD]]

__float128 f128UPreInc(__float128 f) {
  return ++f;
}

// CHECK: cir.func{{.*}} @_Z11f128UPreIncg({{.*}}) -> (!cir.f128{{.*}})
// CHECK:   %[[F128_F:.*]] = cir.alloca !cir.f128, !cir.ptr<!cir.f128>, ["f", init]
// CHECK:   %[[F128_LOAD:.*]] = cir.load{{.*}} %[[F128_F]]
// CHECK:   %[[F128_INC:.*]] = cir.inc %[[F128_LOAD]] : !cir.f128

// LLVM: define{{.*}} fp128 @_Z11f128UPreIncg({{.*}})
// LLVM:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16
// LLVM:   %[[F128_INC:.*]] = fadd fp128 0xL00000000000000003FFF000000000000, %[[F128_LOAD]]

// OGCG: define{{.*}} fp128 @_Z11f128UPreIncg({{.*}})
// OGCG:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16
// OGCG:   %[[F128_INC:.*]] = fadd fp128 %[[F128_LOAD]], 0xL00000000000000003FFF000000000000

__float128 f128UPreDec(__float128 f) {
  return --f;
}

// CHECK: cir.func{{.*}} @_Z11f128UPreDecg({{.*}}) -> (!cir.f128{{.*}})
// CHECK:   %[[F128_F:.*]] = cir.alloca !cir.f128, !cir.ptr<!cir.f128>, ["f", init]
// CHECK:   %[[F128_LOAD:.*]] = cir.load{{.*}} %[[F128_F]]
// CHECK:   %[[F128_DEC:.*]] = cir.dec %[[F128_LOAD]] : !cir.f128

// LLVM: define{{.*}} fp128 @_Z11f128UPreDecg({{.*}})
// LLVM:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16
// LLVM:   %[[F128_DEC:.*]] = fadd fp128 0xL0000000000000000BFFF000000000000, %[[F128_LOAD]]

// OGCG: define{{.*}} fp128 @_Z11f128UPreDecg({{.*}})
// OGCG:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16
// OGCG:   %[[F128_DEC:.*]] = fadd fp128 %[[F128_LOAD]], 0xL0000000000000000BFFF000000000000

__float128 f128UPostInc(__float128 f) {
  return f++;
}

// CHECK: cir.func{{.*}} @_Z12f128UPostIncg({{.*}}) -> (!cir.f128{{.*}})
// CHECK:   %[[F128_F:.*]] = cir.alloca !cir.f128, !cir.ptr<!cir.f128>, ["f", init]
// CHECK:   %[[F128_LOAD:.*]] = cir.load{{.*}} %[[F128_F]]
// CHECK:   %[[F128_INC:.*]] = cir.inc %[[F128_LOAD]] : !cir.f128

// LLVM: define{{.*}} fp128 @_Z12f128UPostIncg({{.*}})
// LLVM:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16
// LLVM:   %[[F128_INC:.*]] = fadd fp128 0xL00000000000000003FFF000000000000, %[[F128_LOAD]]

// OGCG: define{{.*}} fp128 @_Z12f128UPostIncg({{.*}})
// OGCG:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16
// OGCG:   %[[F128_INC:.*]] = fadd fp128 %[[F128_LOAD]], 0xL00000000000000003FFF000000000000

__float128 f128UPostDec(__float128 f) {
  return f--;
}

// CHECK: cir.func{{.*}} @_Z12f128UPostDecg({{.*}}) -> (!cir.f128{{.*}})
// CHECK:   %[[F128_F:.*]] = cir.alloca !cir.f128, !cir.ptr<!cir.f128>, ["f", init]
// CHECK:   %[[F128_LOAD:.*]] = cir.load{{.*}} %[[F128_F]]
// CHECK:   %[[F128_DEC:.*]] = cir.dec %[[F128_LOAD]] : !cir.f128

// LLVM: define{{.*}} fp128 @_Z12f128UPostDecg({{.*}})
// LLVM:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16
// LLVM:   %[[F128_DEC:.*]] = fadd fp128 0xL0000000000000000BFFF000000000000, %[[F128_LOAD]]

// OGCG: define{{.*}} fp128 @_Z12f128UPostDecg({{.*}})
// OGCG:   %[[F128_LOAD:.*]] = load fp128, ptr %{{.*}}, align 16
// OGCG:   %[[F128_DEC:.*]] = fadd fp128 %[[F128_LOAD]], 0xL0000000000000000BFFF000000000000

// Float16 unary operations
_Float16 Float16UPlus(_Float16 f) {
  return +f;
}

// CHECK: cir.func{{.*}} @_Z12Float16UPlusDF16_({{.*}}) -> (!cir.f16{{.*}})
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[F:.*]]
// CHECK:   %[[PROMOTED:.*]] = cir.cast floating %[[INPUT]] : !cir.f16 -> !cir.float
// CHECK:   %[[UNPROMOTED:.*]] = cir.cast floating %[[PROMOTED]] : !cir.float -> !cir.f16

// LLVM: define{{.*}} half @_Z12Float16UPlusDF16_({{.*}})
// LLVM:   %[[F_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// LLVM:   %[[PROMOTED:.*]] = fpext half %[[F_LOAD]] to float
// LLVM:   %[[UNPROMOTED:.*]] = fptrunc float %[[PROMOTED]] to half

// OGCG: define{{.*}} half @_Z12Float16UPlusDF16_({{.*}})
// OGCG:   %[[F_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// OGCG:   %[[PROMOTED:.*]] = fpext half %[[F_LOAD]] to float
// OGCG:   %[[UNPROMOTED:.*]] = fptrunc float %[[PROMOTED]] to half

_Float16 Float16UMinus(_Float16 f) {
  return -f;
}

// CHECK: cir.func{{.*}} @_Z13Float16UMinusDF16_({{.*}}) -> (!cir.f16{{.*}})
// CHECK:   %[[INPUT:.*]] = cir.load{{.*}} %[[F:.*]]
// CHECK:   %[[PROMOTED:.*]] = cir.cast floating %[[INPUT]] : !cir.f16 -> !cir.float
// CHECK:   %[[RESULT:.*]] = cir.minus %[[PROMOTED]]
// CHECK:   %[[UNPROMOTED:.*]] = cir.cast floating %[[RESULT]] : !cir.float -> !cir.f16

// LLVM: define{{.*}} half @_Z13Float16UMinusDF16_({{.*}})
// LLVM:   %[[F_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// LLVM:   %[[PROMOTED:.*]] = fpext half %[[F_LOAD]] to float
// LLVM:   %[[RESULT:.*]] = fneg float %[[PROMOTED]]
// LLVM:   %[[UNPROMOTED:.*]] = fptrunc float %[[RESULT]] to half

// OGCG: define{{.*}} half @_Z13Float16UMinusDF16_({{.*}})
// OGCG:   %[[F_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// OGCG:   %[[PROMOTED:.*]] = fpext half %[[F_LOAD]] to float
// OGCG:   %[[RESULT:.*]] = fneg float %[[PROMOTED]]
// OGCG:   %[[UNPROMOTED:.*]] = fptrunc float %[[RESULT]] to half

_Float16 Float16UPreInc(_Float16 f) {
  return ++f;
}

// CHECK: cir.func{{.*}} @_Z14Float16UPreIncDF16_({{.*}}) -> (!cir.f16{{.*}})
// CHECK:   %[[PREINC_F:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["f", init]
// CHECK:   %[[PREINC_INPUT:.*]] = cir.load{{.*}} %[[PREINC_F]]
// CHECK:   %[[PREINC_RESULT:.*]] = cir.inc %[[PREINC_INPUT]] : !cir.f16

// LLVM: define{{.*}} half @_Z14Float16UPreIncDF16_({{.*}})
// LLVM:   %[[PREINC_F_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM:   %[[PREINC_INPUT:.*]] = load half, ptr %[[PREINC_F_ADDR]], align 2
// LLVM:   %[[PREINC_RESULT:.*]] = fadd half 0xH3C00, %[[PREINC_INPUT]]

// OGCG: define{{.*}} half @_Z14Float16UPreIncDF16_({{.*}})
// OGCG:   %[[PREINC_F_ADDR:.*]] = alloca half, align 2
// OGCG:   %[[PREINC_INPUT:.*]] = load half, ptr %[[PREINC_F_ADDR]], align 2
// OGCG:   %[[PREINC_RESULT:.*]] = fadd half %[[PREINC_INPUT]], 0xH3C00

_Float16 Float16UPreDec(_Float16 f) {
  return --f;
}

// CHECK: cir.func{{.*}} @_Z14Float16UPreDecDF16_({{.*}}) -> (!cir.f16{{.*}})
// CHECK:   %[[PREDEC_F:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["f", init]
// CHECK:   %[[PREDEC_INPUT:.*]] = cir.load{{.*}} %[[PREDEC_F]]
// CHECK:   %[[PREDEC_RESULT:.*]] = cir.dec %[[PREDEC_INPUT]] : !cir.f16

// LLVM: define{{.*}} half @_Z14Float16UPreDecDF16_({{.*}})
// LLVM:   %[[PREDEC_F_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM:   %[[PREDEC_INPUT:.*]] = load half, ptr %[[PREDEC_F_ADDR]], align 2
// LLVM:   %[[PREDEC_RESULT:.*]] = fadd half 0xHBC00, %[[PREDEC_INPUT]]

// OGCG: define{{.*}} half @_Z14Float16UPreDecDF16_({{.*}})
// OGCG:   %[[PREDEC_F_ADDR:.*]] = alloca half, align 2
// OGCG:   %[[PREDEC_INPUT:.*]] = load half, ptr %[[PREDEC_F_ADDR]], align 2
// OGCG:   %[[PREDEC_RESULT:.*]] = fadd half %[[PREDEC_INPUT]], 0xHBC00

_Float16 Float16UPostInc(_Float16 f) {
  return f++;
}

// CHECK: cir.func{{.*}} @_Z15Float16UPostIncDF16_({{.*}}) -> (!cir.f16{{.*}})
// CHECK:   %[[POSTINC_F:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["f", init]
// CHECK:   %[[POSTINC_INPUT:.*]] = cir.load{{.*}} %[[POSTINC_F]]
// CHECK:   %[[POSTINC_RESULT:.*]] = cir.inc %[[POSTINC_INPUT]] : !cir.f16

// LLVM: define{{.*}} half @_Z15Float16UPostIncDF16_({{.*}})
// LLVM:   %[[POSTINC_F_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM:   %[[POSTINC_INPUT:.*]] = load half, ptr %[[POSTINC_F_ADDR]], align 2
// LLVM:   %[[POSTINC_RESULT:.*]] = fadd half 0xH3C00, %[[POSTINC_INPUT]]

// OGCG: define{{.*}} half @_Z15Float16UPostIncDF16_({{.*}})
// OGCG:   %[[POSTINC_F_ADDR:.*]] = alloca half, align 2
// OGCG:   %[[POSTINC_INPUT:.*]] = load half, ptr %[[POSTINC_F_ADDR]], align 2
// OGCG:   %[[POSTINC_RESULT:.*]] = fadd half %[[POSTINC_INPUT]], 0xH3C00

_Float16 Float16UPostDec(_Float16 f) {
  return f--;
}

// CHECK: cir.func{{.*}} @_Z15Float16UPostDecDF16_({{.*}}) -> (!cir.f16{{.*}})
// CHECK:   %[[POSTDEC_F:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["f", init]
// CHECK:   %[[POSTDEC_INPUT:.*]] = cir.load{{.*}} %[[POSTDEC_F]]
// CHECK:   %[[POSTDEC_RESULT:.*]] = cir.dec %[[POSTDEC_INPUT]] : !cir.f16

// LLVM: define{{.*}} half @_Z15Float16UPostDecDF16_({{.*}})
// LLVM:   %[[POSTDEC_F_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM:   %[[POSTDEC_INPUT:.*]] = load half, ptr %[[POSTDEC_F_ADDR]], align 2
// LLVM:   %[[POSTDEC_RESULT:.*]] = fadd half 0xHBC00, %[[POSTDEC_INPUT]]

// OGCG: define{{.*}} half @_Z15Float16UPostDecDF16_({{.*}})
// OGCG:   %[[POSTDEC_F_ADDR:.*]] = alloca half, align 2
// OGCG:   %[[POSTDEC_INPUT:.*]] = load half, ptr %[[POSTDEC_F_ADDR]], align 2
// OGCG:   %[[POSTDEC_RESULT:.*]] = fadd half %[[POSTDEC_INPUT]], 0xHBC00

// __fp16 unary operations
void fp16PtrUPlus(__fp16 *f) {
  *f = +(*f);
}

// CHECK: cir.func{{.*}} @_Z12fp16PtrUPlusPDh({{.*}})
// CHECK:   %[[FPTR_F:.*]] = cir.alloca !cir.ptr<!cir.f16>, !cir.ptr<!cir.ptr<!cir.f16>>, ["f", init]
// CHECK:   %[[FPTR_DEREF:.*]] = cir.load deref{{.*}} %[[FPTR_F]]
// CHECK:   %[[FPTR_LOAD:.*]] = cir.load{{.*}} %[[FPTR_DEREF]]
// CHECK:   %[[FPTR_PROMOTED:.*]] = cir.cast floating %[[FPTR_LOAD]] : !cir.f16 -> !cir.float
// CHECK:   %[[FPTR_UNPROMOTED:.*]] = cir.cast floating %[[FPTR_PROMOTED]] : !cir.float -> !cir.f16

// LLVM: define{{.*}} void @_Z12fp16PtrUPlusPDh({{.*}})
// LLVM:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// LLVM:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// LLVM:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_PROMOTED]] to half

// OGCG: define{{.*}} void @_Z12fp16PtrUPlusPDh({{.*}})
// OGCG:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// OGCG:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// OGCG:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_PROMOTED]] to half

void fp16PtrUMinus(__fp16 *f) {
  *f = -(*f);
}

// CHECK: cir.func{{.*}} @_Z13fp16PtrUMinusPDh({{.*}})
// CHECK:   %[[FPTR_F:.*]] = cir.alloca !cir.ptr<!cir.f16>, !cir.ptr<!cir.ptr<!cir.f16>>, ["f", init]
// CHECK:   %[[FPTR_DEREF:.*]] = cir.load deref{{.*}} %[[FPTR_F]]
// CHECK:   %[[FPTR_LOAD:.*]] = cir.load{{.*}} %[[FPTR_DEREF]]
// CHECK:   %[[FPTR_PROMOTED:.*]] = cir.cast floating %[[FPTR_LOAD]] : !cir.f16 -> !cir.float
// CHECK:   %[[FPTR_NEGATED:.*]] = cir.minus %[[FPTR_PROMOTED]]
// CHECK:   %[[FPTR_UNPROMOTED:.*]] = cir.cast floating %[[FPTR_NEGATED]] : !cir.float -> !cir.f16

// LLVM: define{{.*}} void @_Z13fp16PtrUMinusPDh({{.*}})
// LLVM:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// LLVM:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// LLVM:   %[[FPTR_NEGATED:.*]] = fneg float %[[FPTR_PROMOTED]]
// LLVM:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_NEGATED]] to half

// OGCG: define{{.*}} void @_Z13fp16PtrUMinusPDh({{.*}})
// OGCG:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// OGCG:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// OGCG:   %[[FPTR_NEGATED:.*]] = fneg float %[[FPTR_PROMOTED]]
// OGCG:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_NEGATED]] to half

void fp16PtrUPreInc(__fp16 *f) {
 ++(*f);
}

// CHECK: cir.func{{.*}} @_Z14fp16PtrUPreIncPDh({{.*}})
// CHECK:   %[[FPTR_F:.*]] = cir.alloca !cir.ptr<!cir.f16>, !cir.ptr<!cir.ptr<!cir.f16>>, ["f", init]
// CHECK:   %[[FPTR_DEREF:.*]] = cir.load deref{{.*}} %[[FPTR_F]]
// CHECK:   %[[FPTR_LOAD:.*]] = cir.load{{.*}} %[[FPTR_DEREF]]
// CHECK:   %[[FPTR_PROMOTED:.*]] = cir.cast floating %[[FPTR_LOAD]] : !cir.f16 -> !cir.float
// CHECK:   %[[FPTR_INC:.*]] = cir.inc %[[FPTR_PROMOTED]] : !cir.float
// CHECK:   %[[FPTR_UNPROMOTED:.*]] = cir.cast floating %[[FPTR_INC]] : !cir.float -> !cir.f16

// LLVM: define{{.*}} void @_Z14fp16PtrUPreIncPDh({{.*}})
// LLVM:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// LLVM:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// LLVM:   %[[FPTR_INC:.*]] = fadd float 1.000000e+00, %[[FPTR_PROMOTED]]
// LLVM:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_INC]] to half

// OGCG: define{{.*}} void @_Z14fp16PtrUPreIncPDh({{.*}})
// OGCG:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// OGCG:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// OGCG:   %[[FPTR_INC:.*]] = fadd float %[[FPTR_PROMOTED]], 1.000000e+00
// OGCG:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_INC]] to half

void fp16PtrUPreDec(__fp16 *f) {
  --(*f);
}

// CHECK: cir.func{{.*}} @_Z14fp16PtrUPreDecPDh({{.*}})
// CHECK:   %[[FPTR_F:.*]] = cir.alloca !cir.ptr<!cir.f16>, !cir.ptr<!cir.ptr<!cir.f16>>, ["f", init]
// CHECK:   %[[FPTR_DEREF:.*]] = cir.load deref{{.*}} %[[FPTR_F]]
// CHECK:   %[[FPTR_LOAD:.*]] = cir.load{{.*}} %[[FPTR_DEREF]]
// CHECK:   %[[FPTR_PROMOTED:.*]] = cir.cast floating %[[FPTR_LOAD]] : !cir.f16 -> !cir.float
// CHECK:   %[[FPTR_DEC:.*]] = cir.dec %[[FPTR_PROMOTED]] : !cir.float
// CHECK:   %[[FPTR_UNPROMOTED:.*]] = cir.cast floating %[[FPTR_DEC]] : !cir.float -> !cir.f16

// LLVM: define{{.*}} void @_Z14fp16PtrUPreDecPDh({{.*}})
// LLVM:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// LLVM:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// LLVM:   %[[FPTR_DEC:.*]] = fadd float -1.000000e+00, %[[FPTR_PROMOTED]]
// LLVM:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_DEC]] to half

// OGCG: define{{.*}} void @_Z14fp16PtrUPreDecPDh({{.*}})
// OGCG:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// OGCG:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// OGCG:   %[[FPTR_DEC:.*]] = fadd float %[[FPTR_PROMOTED]], -1.000000e+00
// OGCG:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_DEC]] to half

void fp16PtrUPostInc(__fp16 *f) {
  (*f)++;
}

// CHECK: cir.func{{.*}} @_Z15fp16PtrUPostIncPDh({{.*}})
// CHECK:   %[[FPTR_F:.*]] = cir.alloca !cir.ptr<!cir.f16>, !cir.ptr<!cir.ptr<!cir.f16>>, ["f", init]
// CHECK:   %[[FPTR_DEREF:.*]] = cir.load deref{{.*}} %[[FPTR_F]]
// CHECK:   %[[FPTR_LOAD:.*]] = cir.load{{.*}} %[[FPTR_DEREF]]
// CHECK:   %[[FPTR_PROMOTED:.*]] = cir.cast floating %[[FPTR_LOAD]] : !cir.f16 -> !cir.float
// CHECK:   %[[FPTR_INC:.*]] = cir.inc %[[FPTR_PROMOTED]] : !cir.float
// CHECK:   %[[FPTR_UNPROMOTED:.*]] = cir.cast floating %[[FPTR_INC]] : !cir.float -> !cir.f16

// LLVM: define{{.*}} void @_Z15fp16PtrUPostIncPDh({{.*}})
// LLVM:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// LLVM:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// LLVM:   %[[FPTR_INC:.*]] = fadd float 1.000000e+00, %[[FPTR_PROMOTED]]
// LLVM:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_INC]] to half

// OGCG: define{{.*}} void @_Z15fp16PtrUPostIncPDh({{.*}})
// OGCG:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// OGCG:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// OGCG:   %[[FPTR_INC:.*]] = fadd float %[[FPTR_PROMOTED]], 1.000000e+00
// OGCG:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_INC]] to half

void fp16PtrUPostDec(__fp16 *f) {
  (*f)--;
}

// CHECK: cir.func{{.*}} @_Z15fp16PtrUPostDecPDh({{.*}})
// CHECK:   %[[FPTR_F:.*]] = cir.alloca !cir.ptr<!cir.f16>, !cir.ptr<!cir.ptr<!cir.f16>>, ["f", init]
// CHECK:   %[[FPTR_DEREF:.*]] = cir.load deref{{.*}} %[[FPTR_F]]
// CHECK:   %[[FPTR_LOAD:.*]] = cir.load{{.*}} %[[FPTR_DEREF]]
// CHECK:   %[[FPTR_PROMOTED:.*]] = cir.cast floating %[[FPTR_LOAD]] : !cir.f16 -> !cir.float
// CHECK:   %[[FPTR_DEC:.*]] = cir.dec %[[FPTR_PROMOTED]] : !cir.float
// CHECK:   %[[FPTR_UNPROMOTED:.*]] = cir.cast floating %[[FPTR_DEC]] : !cir.float -> !cir.f16

// LLVM: define{{.*}} void @_Z15fp16PtrUPostDecPDh({{.*}})
// LLVM:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// LLVM:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// LLVM:   %[[FPTR_DEC:.*]] = fadd float -1.000000e+00, %[[FPTR_PROMOTED]]
// LLVM:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_DEC]] to half

// OGCG: define{{.*}} void @_Z15fp16PtrUPostDecPDh({{.*}})
// OGCG:   %[[FPTR_LOAD:.*]] = load half, ptr %{{.*}}, align 2
// OGCG:   %[[FPTR_PROMOTED:.*]] = fpext half %[[FPTR_LOAD]] to float
// OGCG:   %[[FPTR_DEC:.*]] = fadd float %[[FPTR_PROMOTED]], -1.000000e+00
// OGCG:   %[[FPTR_UNPROMOTED:.*]] = fptrunc float %[[FPTR_DEC]] to half

// __bf16 unary operations
__bf16 bf16UPlus(__bf16 f) {
  return +f;
}

// CHECK: cir.func{{.*}} @_Z9bf16UPlusDF16b({{.*}}) -> (!cir.bf16{{.*}})
// CHECK:   %[[BF16_F:.*]] = cir.alloca !cir.bf16, !cir.ptr<!cir.bf16>, ["f", init]
// CHECK:   %[[BF16_LOAD:.*]] = cir.load{{.*}} %[[BF16_F]]
// CHECK:   %[[BF16_PROMOTED:.*]] = cir.cast floating %[[BF16_LOAD]] : !cir.bf16 -> !cir.float
// CHECK:   %[[BF16_UNPROMOTED:.*]] = cir.cast floating %[[BF16_PROMOTED]] : !cir.float -> !cir.bf16

// LLVM: define{{.*}} bfloat @_Z9bf16UPlusDF16b({{.*}})
// LLVM:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// LLVM:   %[[BF16_PROMOTED:.*]] = fpext bfloat %[[BF16_LOAD]] to float
// LLVM:   %[[BF16_UNPROMOTED:.*]] = fptrunc float %[[BF16_PROMOTED]] to bfloat

// OGCG: define{{.*}} bfloat @_Z9bf16UPlusDF16b({{.*}})
// OGCG:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// OGCG:   %[[BF16_PROMOTED:.*]] = fpext bfloat %[[BF16_LOAD]] to float
// OGCG:   %[[BF16_UNPROMOTED:.*]] = fptrunc float %[[BF16_PROMOTED]] to bfloat

__bf16 bf16UMinus(__bf16 f) {
  return -f;
}

// CHECK: cir.func{{.*}} @_Z10bf16UMinusDF16b({{.*}}) -> (!cir.bf16{{.*}})
// CHECK:   %[[BF16_F:.*]] = cir.alloca !cir.bf16, !cir.ptr<!cir.bf16>, ["f", init]
// CHECK:   %[[BF16_LOAD:.*]] = cir.load{{.*}} %[[BF16_F]]
// CHECK:   %[[BF16_PROMOTED:.*]] = cir.cast floating %[[BF16_LOAD]] : !cir.bf16 -> !cir.float
// CHECK:   %[[BF16_NEGATED:.*]] = cir.minus %[[BF16_PROMOTED]]
// CHECK:   %[[BF16_UNPROMOTED:.*]] = cir.cast floating %[[BF16_NEGATED]] : !cir.float -> !cir.bf16

// LLVM: define{{.*}} bfloat @_Z10bf16UMinusDF16b({{.*}})
// LLVM:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// LLVM:   %[[BF16_PROMOTED:.*]] = fpext bfloat %[[BF16_LOAD]] to float
// LLVM:   %[[BF16_NEGATED:.*]] = fneg float %[[BF16_PROMOTED]]
// LLVM:   %[[BF16_UNPROMOTED:.*]] = fptrunc float %[[BF16_NEGATED]] to bfloat

// OGCG: define{{.*}} bfloat @_Z10bf16UMinusDF16b({{.*}})
// OGCG:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// OGCG:   %[[BF16_PROMOTED:.*]] = fpext bfloat %[[BF16_LOAD]] to float
// OGCG:   %[[BF16_NEGATED:.*]] = fneg float %[[BF16_PROMOTED]]
// OGCG:   %[[BF16_UNPROMOTED:.*]] = fptrunc float %[[BF16_NEGATED]] to bfloat

__bf16 bf16UPreInc(__bf16 f) {
  return ++f;
}

// CHECK: cir.func{{.*}} @_Z11bf16UPreIncDF16b({{.*}}) -> (!cir.bf16{{.*}})
// CHECK:   %[[BF16_F:.*]] = cir.alloca !cir.bf16, !cir.ptr<!cir.bf16>, ["f", init]
// CHECK:   %[[BF16_LOAD:.*]] = cir.load{{.*}} %[[BF16_F]]
// CHECK:   %[[BF16_INC:.*]] = cir.inc %[[BF16_LOAD]] : !cir.bf16

// LLVM: define{{.*}} bfloat @_Z11bf16UPreIncDF16b({{.*}})
// LLVM:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// LLVM:   %[[BF16_INC:.*]] = fadd bfloat 0xR3F80, %[[BF16_LOAD]]

// OGCG: define{{.*}} bfloat @_Z11bf16UPreIncDF16b({{.*}})
// OGCG:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// OGCG:   %[[BF16_INC:.*]] = fadd bfloat %[[BF16_LOAD]], 0xR3F80

__bf16 bf16UPreDec(__bf16 f) {
  return --f;
}

// CHECK: cir.func{{.*}} @_Z11bf16UPreDecDF16b({{.*}}) -> (!cir.bf16{{.*}})
// CHECK:   %[[BF16_F:.*]] = cir.alloca !cir.bf16, !cir.ptr<!cir.bf16>, ["f", init]
// CHECK:   %[[BF16_LOAD:.*]] = cir.load{{.*}} %[[BF16_F]]
// CHECK:   %[[BF16_DEC:.*]] = cir.dec %[[BF16_LOAD]] : !cir.bf16

// LLVM: define{{.*}} bfloat @_Z11bf16UPreDecDF16b({{.*}})
// LLVM:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// LLVM:   %[[BF16_DEC:.*]] = fadd bfloat 0xRBF80, %[[BF16_LOAD]]

// OGCG: define{{.*}} bfloat @_Z11bf16UPreDecDF16b({{.*}})
// OGCG:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// OGCG:   %[[BF16_DEC:.*]] = fadd bfloat %[[BF16_LOAD]], 0xRBF80

__bf16 bf16UPostInc(__bf16 f) {
  return f++;
}

// CHECK: cir.func{{.*}} @_Z12bf16UPostIncDF16b({{.*}}) -> (!cir.bf16{{.*}})
// CHECK:   %[[BF16_F:.*]] = cir.alloca !cir.bf16, !cir.ptr<!cir.bf16>, ["f", init]
// CHECK:   %[[BF16_LOAD:.*]] = cir.load{{.*}} %[[BF16_F]]
// CHECK:   %[[BF16_INC:.*]] = cir.inc %[[BF16_LOAD]] : !cir.bf16

// LLVM: define{{.*}} bfloat @_Z12bf16UPostIncDF16b({{.*}})
// LLVM:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// LLVM:   %[[BF16_INC:.*]] = fadd bfloat 0xR3F80, %[[BF16_LOAD]]

// OGCG: define{{.*}} bfloat @_Z12bf16UPostIncDF16b({{.*}})
// OGCG:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// OGCG:   %[[BF16_INC:.*]] = fadd bfloat %[[BF16_LOAD]], 0xR3F80

__bf16 bf16UPostDec(__bf16 f) {
  return f--;
}

// CHECK: cir.func{{.*}} @_Z12bf16UPostDecDF16b({{.*}}) -> (!cir.bf16{{.*}})
// CHECK:   %[[BF16_F:.*]] = cir.alloca !cir.bf16, !cir.ptr<!cir.bf16>, ["f", init]
// CHECK:   %[[BF16_LOAD:.*]] = cir.load{{.*}} %[[BF16_F]]
// CHECK:   %[[BF16_DEC:.*]] = cir.dec %[[BF16_LOAD]] : !cir.bf16

// LLVM: define{{.*}} bfloat @_Z12bf16UPostDecDF16b({{.*}})
// LLVM:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// LLVM:   %[[BF16_DEC:.*]] = fadd bfloat 0xRBF80, %[[BF16_LOAD]]

// OGCG: define{{.*}} bfloat @_Z12bf16UPostDecDF16b({{.*}})
// OGCG:   %[[BF16_LOAD:.*]] = load bfloat, ptr %{{.*}}, align 2
// OGCG:   %[[BF16_DEC:.*]] = fadd bfloat %[[BF16_LOAD]], 0xRBF80

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

// CHECK: cir.func{{.*}} @_Z16test_logical_notv()
// CHECK:   %[[A:.*]] = cir.load{{.*}} %[[A_ADDR:.*]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[A_BOOL:.*]] = cir.cast int_to_bool %[[A]] : !s32i -> !cir.bool
// CHECK:   %[[A_NOT:.*]] = cir.not %[[A_BOOL]] : !cir.bool
// CHECK:   %[[A_CAST:.*]] = cir.cast bool_to_int %[[A_NOT]] : !cir.bool -> !s32i
// CHECK:   cir.store{{.*}} %[[A_CAST]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[B:.*]] = cir.load{{.*}} %[[B_ADDR:.*]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:   %[[B_NOT:.*]] = cir.not %[[B]] : !cir.bool
// CHECK:   cir.store{{.*}} %[[B_NOT]], %[[B_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:   %[[C:.*]] = cir.load{{.*}} %[[C_ADDR:.*]] : !cir.ptr<!cir.float>, !cir.float
// CHECK:   %[[C_BOOL:.*]] = cir.cast float_to_bool %[[C]] : !cir.float -> !cir.bool
// CHECK:   %[[C_NOT:.*]] = cir.not %[[C_BOOL]] : !cir.bool
// CHECK:   %[[C_CAST:.*]] = cir.cast bool_to_float %[[C_NOT]] : !cir.bool -> !cir.float
// CHECK:   cir.store{{.*}} %[[C_CAST]], %[[C_ADDR]] : !cir.float, !cir.ptr<!cir.float>
// CHECK:   %[[P:.*]] = cir.load{{.*}} %[[P_ADDR:.*]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:   %[[P_BOOL:.*]] = cir.cast ptr_to_bool %[[P]] : !cir.ptr<!s32i> -> !cir.bool
// CHECK:   %[[P_NOT:.*]] = cir.not %[[P_BOOL]] : !cir.bool
// CHECK:   cir.store{{.*}} %[[P_NOT]], %[[B_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:   %[[D:.*]] = cir.load{{.*}} %[[D_ADDR:.*]] : !cir.ptr<!cir.double>, !cir.double
// CHECK:   %[[D_BOOL:.*]] = cir.cast float_to_bool %[[D]] : !cir.double -> !cir.bool
// CHECK:   %[[D_NOT:.*]] = cir.not %[[D_BOOL]] : !cir.bool
// CHECK:   cir.store{{.*}} %[[D_NOT]], %[[B_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: define{{.*}} void @_Z16test_logical_notv()
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
// OGCG:   %[[B_BOOL:.*]] = icmp ne i8 %[[B]], 0
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

void f16NestedUPlus() {
  _Float16 a;
  _Float16 b = +(+a);
}

// CHECK: cir.func{{.*}} @_Z14f16NestedUPlusv()
// CHECK:  %[[A_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["a"]
// CHECK:  %[[B_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["b", init]
// CHECK:  %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.f16>, !cir.f16
// CHECK:  %[[A_F32:.*]] = cir.cast floating %[[TMP_A]] : !cir.f16 -> !cir.float
// CHECK:  %[[RESULT:.*]] = cir.cast floating %[[A_F32]] : !cir.float -> !cir.f16
// CHECK:  cir.store{{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.f16, !cir.ptr<!cir.f16>

// LLVM: define{{.*}} void @_Z14f16NestedUPlusv()
// LLVM:  %[[A_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM:  %[[B_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM:  %[[TMP_A:.*]] = load half, ptr %[[A_ADDR]], align 2
// LLVM:  %[[RESULT_F32:.*]] = fpext half %[[TMP_A]] to float
// LLVM:  %[[RESULT:.*]] = fptrunc float %[[RESULT_F32]] to half
// LLVM:  store half %[[RESULT]], ptr %[[B_ADDR]], align 2

// OGCG: define{{.*}} void @_Z14f16NestedUPlusv()
// OGCG:  %[[A_ADDR:.*]] = alloca half, align 2
// OGCG:  %[[B_ADDR:.*]] = alloca half, align 2
// OGCG:  %[[TMP_A:.*]] = load half, ptr %[[A_ADDR]], align 2
// OGCG:  %[[RESULT_F32:.*]] = fpext half %[[TMP_A]] to float
// OGCG:  %[[RESULT:.*]] = fptrunc float %[[RESULT_F32]] to half
// OGCG:  store half %[[RESULT]], ptr %[[B_ADDR]], align 2

void f16NestedUMinus() {
  _Float16 a;
  _Float16 b = -(-a);
}

// CHECK: cir.func{{.*}} @_Z15f16NestedUMinusv()
// CHECK:  %[[A_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["a"]
// CHECK:  %[[B_ADDR:.*]] = cir.alloca !cir.f16, !cir.ptr<!cir.f16>, ["b", init]
// CHECK:  %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.f16>, !cir.f16
// CHECK:  %[[A_F32:.*]] = cir.cast floating %[[TMP_A]] : !cir.f16 -> !cir.float
// CHECK:  %[[A_MINUS:.*]] = cir.minus %[[A_F32]] : !cir.float
// CHECK:  %[[RESULT_F32:.*]] = cir.minus %[[A_MINUS]] : !cir.float
// CHECK:  %[[RESULT:.*]] = cir.cast floating %[[RESULT_F32]] : !cir.float -> !cir.f16
// CHECK:  cir.store{{.*}} %[[RESULT]], %[[B_ADDR]] : !cir.f16, !cir.ptr<!cir.f16>

// LLVM: define{{.*}} void @_Z15f16NestedUMinusv()
// LLVM:  %[[A_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM:  %[[B_ADDR:.*]] = alloca half, i64 1, align 2
// LLVM:  %[[TMP_A:.*]] = load half, ptr %[[A_ADDR]], align 2
// LLVM:  %[[A_F32:.*]] = fpext half %[[TMP_A]] to float
// LLVM:  %[[A_MINUS:.*]] = fneg float %[[A_F32]]
// LLVM:  %[[RESULT_F32:.*]] = fneg float %[[A_MINUS]]
// LLVM:  %[[RESULT:.*]] = fptrunc float %[[RESULT_F32]] to half
// LLVM:  store half %[[RESULT]], ptr %[[B_ADDR]], align 2

// OGCG: define{{.*}} void @_Z15f16NestedUMinusv()
// OGCG:  %[[A_ADDR:.*]] = alloca half, align 2
// OGCG:  %[[B_ADDR:.*]] = alloca half, align 2
// OGCG:  %[[TMP_A:.*]] = load half, ptr %[[A_ADDR]], align 2
// OGCG:  %[[A_F32:.*]] = fpext half %[[TMP_A]] to float
// OGCG:  %[[A_MINUS:.*]] = fneg float %[[A_F32]]
// OGCG:  %[[RESULT_F32:.*]] = fneg float %[[A_MINUS]]
// OGCG:  %[[RESULT:.*]] = fptrunc float %[[RESULT_F32]] to half
// OGCG:  store half %[[RESULT]], ptr %[[B_ADDR]], align 2
