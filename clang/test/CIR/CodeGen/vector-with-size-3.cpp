// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

typedef int vi3 __attribute__((ext_vector_type(3)));

void store_load() {
  vi3 a;
  vi3 b = a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.vector<3 x !s32i>, !cir.ptr<!cir.vector<3 x !s32i>>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.vector<3 x !s32i>, !cir.ptr<!cir.vector<3 x !s32i>>, ["b", init]
// CIR: %[[A_V4:.*]] = cir.cast(bitcast, %[[A_ADDR]] : !cir.ptr<!cir.vector<3 x !s32i>>), !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A_V4:.*]] = cir.load{{.*}} %[[A_V4]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<4 x !s32i>
// CIR: %[[SHUFFLE_V4:.*]] = cir.vec.shuffle(%3, %[[POISON]] : !cir.vector<4 x !s32i>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i] : !cir.vector<3 x !s32i>
// CIR: %[[POISON:.*]] = cir.const #cir.poison : !cir.vector<3 x !s32i>
// CIR: %[[SHUFFLE_V3:.*]] = cir.vec.shuffle(%[[SHUFFLE_V4]], %[[POISON]] : !cir.vector<3 x !s32i>) [#cir.int<0> : !s64i, #cir.int<1> : !s64i, #cir.int<2> : !s64i, #cir.int<-1> : !s64i] : !cir.vector<4 x !s32i>
// CIR: %[[TMP_B_V4:.*]] = cir.cast(bitcast, %[[B_ADDR]] : !cir.ptr<!cir.vector<3 x !s32i>>), !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: cir.store{{.*}} %[[SHUFFLE_V3]], %[[TMP_B_V4]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca <3 x i32>, i64 1, align 16
// LLVM: %[[B_ADDR:.*]] = alloca <3 x i32>, i64 1, align 16
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// LLVM: %[[SHUFFLE_V4:.*]] = shufflevector <4 x i32> %[[TMP_A]], <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
// LLVM: %[[SHUFFLE_V3:.*]] = shufflevector <3 x i32> %[[SHUFFLE_V4]], <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// LLVM: store <4 x i32> %[[SHUFFLE_V3]], ptr %[[B_ADDR]], align 16

// OGCG: %[[A_ADDR:.*]] = alloca <3 x i32>, align 16
// OGCG: %[[B_ADDR:.*]] = alloca <3 x i32>, align 16
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// OGCG: %[[SHUFFLE_V4:.*]] = shufflevector <4 x i32> %[[TMP_A]], <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
// OGCG: %[[SHUFFLE_V3:.*]] = shufflevector <3 x i32> %[[SHUFFLE_V4]], <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// OGCG: store <4 x i32> %[[SHUFFLE_V3]], ptr %[[B_ADDR]], align 16
