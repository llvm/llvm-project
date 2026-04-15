// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

using Arr4Ty = int [4];
using ArrNTy = int [];
using CArrNTy = const int [];

void cast1() {
  Arr4Ty arr = {};
  ArrNTy &toArr = arr;

  // CIR-LABEL: cir.func {{.*}}@_Z5cast1v()
  // CIR: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 4>, !cir.ptr<!cir.array<!s32i x 4>>, ["arr", init]
  // CIR: %[[TO_ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!s32i x 0>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>, ["toArr", init, const]
  // CIR: %[[TO_INCOMPLETE:.*]] = cir.cast bitcast %[[ARR_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!cir.array<!s32i x 0>>
  // CIR: cir.store {{.*}}%[[TO_INCOMPLETE]], %[[TO_ARR_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 0>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>

  // LLVM-LABEL: define {{.*}}@_Z5cast1v()
  // LLVM: %[[ARR_ALLOCA:.*]] = alloca [4 x i32]
  // LLVM: %[[TO_ARR_ALLOCA:.*]] = alloca ptr
  // LLVM: store ptr %[[ARR_ALLOCA]], ptr %[[TO_ARR_ALLOCA]]
}

void cast2() {
  Arr4Ty arr = {};
  CArrNTy &toArr = arr;
  // CIR-LABEL: cir.func {{.*}}@_Z5cast2v()
  // CIR: %[[ARR_ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 4>, !cir.ptr<!cir.array<!s32i x 4>>, ["arr", init]
  // CIR: %[[TO_ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!s32i x 0>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>, ["toArr", init, const]
  // CIR: %[[TO_INCOMPLETE:.*]] = cir.cast bitcast %[[ARR_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!cir.array<!s32i x 0>>
  // CIR: cir.store {{.*}}%[[TO_INCOMPLETE]], %[[TO_ARR_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 0>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>
  //
  // LLVM-LABEL: define {{.*}}@_Z5cast2v()
  // LLVM: %[[ARR_ALLOCA:.*]] = alloca [4 x i32]
  // LLVM: %[[TO_ARR_ALLOCA:.*]] = alloca ptr
  // LLVM: store ptr %[[ARR_ALLOCA]], ptr %[[TO_ARR_ALLOCA]]
}

void cast3() {
  int (&&toArr)[] = static_cast<int[]>(3);
  // CIR-LABEL: cir.func {{.*}}@_Z5cast3v()
  // CIR: %[[TMP_ALLOCA:.*]] = cir.alloca !cir.array<!s32i x 1>, !cir.ptr<!cir.array<!s32i x 1>>, ["ref.tmp0"] 
  // CIR: %[[TO_ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!s32i x 0>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>, ["toArr", init, const]
  // CIR: %[[TO_INCOMPLETE:.*]] = cir.cast bitcast %[[TMP_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 1>> -> !cir.ptr<!cir.array<!s32i x 0>>
  // CIR: cir.store {{.*}}%[[TO_INCOMPLETE]], %[[TO_ARR_ALLOCA]] : !cir.ptr<!cir.array<!s32i x 0>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>
  //
  // LLVM-LABEL: define {{.*}}@_Z5cast3v()
  // LLVM: %[[ARR_ALLOCA:.*]] = alloca [1 x i32]
  // LLVM: %[[TO_ARR_ALLOCA:.*]] = alloca ptr
  // LLVM: store ptr %[[ARR_ALLOCA]], ptr %[[TO_ARR_ALLOCA]]
}

void cast4() {
  Arr4Ty* const *arrPP;
  CArrNTy* const volatile *const constArrPP = arrPP;

  // CIR-LABEL: cir.func {{.*}}@_Z5cast4v()
  // CIR: %[[ARR_PP_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.array<!s32i x 4>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 4>>>>, ["arrPP"]
  // CIR: %[[CONST_ARR_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>>, ["constArrPP", init, const]
  // CIR: %[[LOAD_ARR_PP:.*]] = cir.load align(8) %[[ARR_PP_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 4>>>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 4>>>
  // CIR: %[[CONST_ARR_CAST:.*]] = cir.cast bitcast %[[CONST_ARR_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 0>>>> -> !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 4>>>>
  // CIR: cir.store{{.*}} %[[LOAD_ARR_PP]], %[[CONST_ARR_CAST]] : !cir.ptr<!cir.ptr<!cir.array<!s32i x 4>>>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.array<!s32i x 4>>>>
  //
  // LLVM-LABEL: define {{.*}}@_Z5cast4v()
  // LLVM: %[[ARR_PP_ALLOCA:.*]] = alloca ptr
  // LLVM: %[[CONST_ARR_ALLOCA:.*]] = alloca ptr
  // LLVM: %[[LOAD_ARR_PP:.*]] = load ptr, ptr %[[ARR_PP_ALLOCA]]
  // LLVM: store ptr %[[LOAD_ARR_PP]], ptr %[[CONST_ARR_ALLOCA]]

}
