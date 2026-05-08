// RUN: %clang_cc1 %s -fclangir -emit-cir -triple spir-unknown-unknown -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR

// RUN: %clang_cc1 %s -fclangir -emit-llvm -triple spir-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

// RUN: %clang_cc1 %s -emit-llvm -triple spir-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=OGCG

typedef __attribute__(( ext_vector_type(4) )) char char4;

char4 f4(int x) {
  return __builtin_astype(x, char4);
}

// CIR: cir.func {{.*}} @f4
// CIR:   %[[X_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR:   %[[RET_ADDR:.*]] = cir.alloca !cir.vector<4 x !s8i>, !cir.ptr<!cir.vector<4 x !s8i>>, ["__retval"]
// CIR:   cir.store %{{.*}}, %[[X_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP_X:.*]] = cir.load {{.*}} %[[X_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[X_V4_I8:.*]] = cir.cast bitcast %[[TMP_X]] : !s32i -> !cir.vector<4 x !s8i>
// CIR:   cir.store %[[X_V4_I8]], %[[RET_ADDR]] : !cir.vector<4 x !s8i>, !cir.ptr<!cir.vector<4 x !s8i>>
// CIR:   %[[TMP_RET:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!cir.vector<4 x !s8i>>, !cir.vector<4 x !s8i>
// CIR:   cir.return %[[TMP_RET]] : !cir.vector<4 x !s8i>

// LLVM: define {{.*}} <4 x i8> @f4
// LLVM:  %[[RET:.*]] = bitcast i32 %{{.*}} to <4 x i8>
// LLVM:  ret <4 x i8> %[[RET]]

// OGCG: define {{.*}} <4 x i8> @f4
// OGCG:  %[[RET:.*]] = bitcast i32 %{{.*}} to <4 x i8>
// OGCG:  ret <4 x i8> %[[RET]]

int f6(char4 x) {
  return __builtin_astype(x, int);
}

// CIR: cir.func {{.*}} @f6
// CIR:   %[[X_ADDR:.*]] = cir.alloca !cir.vector<4 x !s8i>, !cir.ptr<!cir.vector<4 x !s8i>>, ["x", init]
// CIR:   %[[RET_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   cir.store %{{.*}}, %[[X_ADDR]] : !cir.vector<4 x !s8i>, !cir.ptr<!cir.vector<4 x !s8i>>
// CIR:   %[[TMP_X:.*]] = cir.load {{.*}} %[[X_ADDR]] : !cir.ptr<!cir.vector<4 x !s8i>>, !cir.vector<4 x !s8i>
// CIR:   %[[X_S32I:.*]] = cir.cast bitcast %[[TMP_X]] : !cir.vector<4 x !s8i> -> !s32i
// CIR:   cir.store %[[X_S32I]], %[[RET_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP_RET:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP_RET]] : !s32i

// LLVM: define {{.*}} i32 @f6
// LLVM:  %[[RET:.*]] = bitcast <4 x i8> %{{.*}} to i32
// LLVM:  ret i32 %[[RET]]

// OGCG: define {{.*}} i32 @f6
// OGCG:  %[[RET:.*]] = bitcast <4 x i8> %{{.*}} to i32
// OGCG:  ret i32 %[[RET]]

int* int_to_ptr(int x) {
  return __builtin_astype(x, int*);
}

// CIR: cir.func {{.*}} @int_to_ptr
// CIR:   %[[X_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR:   %[[RET_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["__retval"]
// CIR:   cir.store %{{.*}}, %[[X_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP_X:.*]] = cir.load {{.*}} %[[X_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[X_PTR:.*]] = cir.cast int_to_ptr %[[TMP_X]] : !s32i -> !cir.ptr<!s32i>
// CIR:   cir.store %[[X_PTR]], %[[RET_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:   %[[TMP_RET]] = cir.load %[[RET_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:   cir.return %[[TMP_RET]] : !cir.ptr<!s32i>

// LLVM: define {{.*}} ptr @int_to_ptr
// LLVM:   %[[INT_TO_PTR:.*]] = inttoptr i32 %{{.*}} to ptr
// LLVM:   ret ptr %[[INT_TO_PTR]]

// OGCG: define {{.*}} ptr @int_to_ptr
// OGCG:   %[[INT_TO_PTR:.*]] = inttoptr i32 %{{.*}} to ptr
// OGCG:   ret ptr %[[INT_TO_PTR]]
