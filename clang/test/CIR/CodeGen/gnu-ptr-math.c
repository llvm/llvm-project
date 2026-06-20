// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM_CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVM_OG

typedef void *vptr;
typedef int(*fptr)(int, double);

vptr vptr_add(vptr p) {
  // CIR-LABEL: vptr_add
  // CIR: %[[ARG:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["p", init]
  // CIR: %[[RET:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["__retval"]
  // CIR: %[[ARG_LOAD:.*]] = cir.load {{.*}}%[[ARG]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR: %[[OFFSET:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: %[[STRIDE:.*]] = cir.ptr_stride %[[ARG_LOAD]], %[[OFFSET]]
  // CIR: cir.store {{.*}}%[[STRIDE]], %[[RET]]

  // LLVM-LABEL: vptr_add
  // LLVM: %[[ARG:.*]] = alloca ptr
  // LLVM_CIR: %[[RET:.*]] = alloca ptr
  // LLVM: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG]]
  // LLVM: %[[STRIDE:.*]] = getelementptr {{.*}}i8, ptr %[[ARG_LOAD]], i64 3
  // LLVM_CIR: store ptr %[[STRIDE:.*]], ptr %[[RET]]
  // LLVM_OG: ret ptr %[[STRIDE]]
  return p + 3;
}

vptr vptr_sub(vptr p) {
  // CIR-LABEL: vptr_sub
  // CIR: %[[ARG:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["p", init]
  // CIR: %[[RET:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["__retval"]
  // CIR: %[[ARG_LOAD:.*]] = cir.load {{.*}}%[[ARG]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR: %[[OFFSET:.*]] = cir.const #cir.int<-2> : !s32i
  // CIR: %[[STRIDE:.*]] = cir.ptr_stride %[[ARG_LOAD]], %[[OFFSET]]
  // CIR: cir.store {{.*}}%[[STRIDE]], %[[RET]]

  // LLVM-LABEL: vptr_sub
  // LLVM: %[[ARG:.*]] = alloca ptr
  // LLVM_CIR: %[[RET:.*]] = alloca ptr
  // LLVM: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG]]
  // LLVM: %[[STRIDE:.*]] = getelementptr {{.*}}i8, ptr %[[ARG_LOAD]], i64 -2
  // LLVM_CIR: store ptr %[[STRIDE:.*]], ptr %[[RET]]
  // LLVM_OG: ret ptr %[[STRIDE]]
  return p - 2;
}

vptr vptr_inc(vptr p) {
  // CIR-LABEL: vptr_inc
  // CIR: %[[ARG:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["p", init]
  // CIR: %[[RET:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["__retval"]
  // CIR: %[[ARG_LOAD:.*]] = cir.load {{.*}}%[[ARG]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[STRIDE:.*]] = cir.ptr_stride %[[ARG_LOAD]], %[[OFFSET]]
  // CIR: cir.store {{.*}}%[[STRIDE]], %[[ARG]]

  // LLVM-LABEL: vptr_inc
  // LLVM: %[[ARG:.*]] = alloca ptr
  // LLVM_CIR: %[[RET:.*]] = alloca ptr
  // LLVM: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG]]
  // LLVM: %[[STRIDE:.*]] = getelementptr {{.*}}i8, ptr %[[ARG_LOAD]], {{.*}} 1
  // LLVM_CIR: store ptr %[[STRIDE:.*]], ptr %[[RET]]
  // LLVM_OG: store ptr %[[STRIDE]], ptr %[[ARG]]
  return p ++;
}
vptr vptr_dec(vptr p) {
  // CIR-LABEL: vptr_dec
  // CIR: %[[ARG:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["p", init]
  // CIR: %[[RET:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["__retval"]
  // CIR: %[[ARG_LOAD:.*]] = cir.load {{.*}}%[[ARG]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR: %[[OFFSET:.*]] = cir.const #cir.int<-1> : !s32i
  // CIR: %[[STRIDE:.*]] = cir.ptr_stride %[[ARG_LOAD]], %[[OFFSET]]
  // CIR: cir.store {{.*}}%[[STRIDE]], %[[RET]]

  // LLVM-LABEL: vptr_dec
  // LLVM: %[[ARG:.*]] = alloca ptr
  // LLVM_CIR: %[[RET:.*]] = alloca ptr
  // LLVM: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG]]
  // LLVM: %[[STRIDE:.*]] = getelementptr {{.*}}i8, ptr %[[ARG_LOAD]], {{.*}} -1
  // LLVM_CIR: store ptr %[[STRIDE:.*]], ptr %[[RET]]
  // LLVM_OG: ret ptr %[[STRIDE]]
  return --p;
}

fptr fptr_add(fptr p) {
  // CIR-LABEL: fptr_add
  // CIR: %[[ARG:.*]] = cir.alloca !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, ["p", init]
  // CIR: %[[RET:.*]] = cir.alloca !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, ["__retval"]
  // CIR: %[[ARG_LOAD:.*]] = cir.load {{.*}}%[[ARG]] : !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>
  // CIR: %[[OFFSET:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: %[[STRIDE:.*]] = cir.ptr_stride %[[ARG_LOAD]], %[[OFFSET]]
  // CIR: cir.store {{.*}}%[[STRIDE]], %[[RET]]

  // LLVM-LABEL: fptr_add
  // LLVM: %[[ARG:.*]] = alloca ptr
  // LLVM-CIR: %[[RET:.*]] = alloca ptr
  // LLVM: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG]]
  // LLVM: %[[STRIDE:.*]] = getelementptr {{.*}}i8, ptr %[[ARG_LOAD]], i64 3
  // LLVM_CIR: store ptr %[[STRIDE:.*]], ptr %[[RET]]
  // LLVM_OG: ret ptr %[[STRIDE]]
  return p + 3;
}

fptr fptr_sub(fptr p) {
  // CIR-LABEL: fptr_sub
  // CIR: %[[ARG:.*]] = cir.alloca !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, ["p", init]
  // CIR: %[[RET:.*]] = cir.alloca !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, ["__retval"]
  // CIR: %[[ARG_LOAD:.*]] = cir.load {{.*}}%[[ARG]] : !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>
  // CIR: %[[OFFSET:.*]] = cir.const #cir.int<-2> : !s32i
  // CIR: %[[STRIDE:.*]] = cir.ptr_stride %[[ARG_LOAD]], %[[OFFSET]]
  // CIR: cir.store {{.*}}%[[STRIDE]], %[[RET]]

  // LLVM-LABEL: fptr_sub
  // LLVM: %[[ARG:.*]] = alloca ptr
  // LLVM_CIR: %[[RET:.*]] = alloca ptr
  // LLVM: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG]]
  // LLVM: %[[STRIDE:.*]] = getelementptr {{.*}}i8, ptr %[[ARG_LOAD]], i64 -2
  // LLVM_CIR: store ptr %[[STRIDE:.*]], ptr %[[RET]]
  // LLVM_OG: ret ptr %[[STRIDE]]
  return p - 2;
}

fptr fptr_inc(fptr p) {
  // CIR-LABEL: fptr_inc
  // CIR: %[[ARG:.*]] = cir.alloca !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, ["p", init]
  // CIR: %[[RET:.*]] = cir.alloca !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, ["__retval"]
  // CIR: %[[ARG_LOAD:.*]] = cir.load {{.*}}%[[ARG]] : !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>
  // CIR: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[STRIDE:.*]] = cir.ptr_stride %[[ARG_LOAD]], %[[OFFSET]]
  // CIR: cir.store {{.*}}%[[STRIDE]], %[[ARG]]

  // LLVM-LABEL: fptr_inc
  // LLVM: %[[ARG:.*]] = alloca ptr
  // LLVM_CIR: %[[RET:.*]] = alloca ptr
  // LLVM: %[[STRIDE:.*]] = getelementptr {{.*}}i8, ptr %[[ARG_LOAD]], {{.*}} 1
  // LLVM_CIR: store ptr %[[STRIDE:.*]], ptr %[[RET]]
  // LLVM_OG: store ptr %[[STRIDE]], ptr %[[ARG]]
  return p ++;
}
fptr fptr_dec(fptr p) {
  // CIR-LABEL: fptr_dec
  // CIR: %[[ARG:.*]] = cir.alloca !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, ["p", init]
  // CIR: %[[RET:.*]] = cir.alloca !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, ["__retval"]
  // CIR: %[[ARG_LOAD:.*]] = cir.load {{.*}}%[[ARG]] : !cir.ptr<!cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>>, !cir.ptr<!cir.func<(!s32i, !cir.double) -> !s32i>>
  // CIR: %[[OFFSET:.*]] = cir.const #cir.int<-1> : !s32i
  // CIR: %[[STRIDE:.*]] = cir.ptr_stride %[[ARG_LOAD]], %[[OFFSET]]
  // CIR: cir.store {{.*}}%[[STRIDE]], %[[RET]]

  // LLVM-LABEL: fptr_dec
  // LLVM: %[[ARG:.*]] = alloca ptr
  // LLVM_CIR: %[[RET:.*]] = alloca ptr
  // LLVM: %[[STRIDE:.*]] = getelementptr {{.*}}i8, ptr %[[ARG_LOAD]], {{.*}} -1
  // LLVM_CIR: store ptr %[[STRIDE:.*]], ptr %[[RET]]
  // LLVM_OG: ret ptr %[[STRIDE]]
  return --p;
}
