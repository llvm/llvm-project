// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct Data {
  int value;
  void *ptr;
};

typedef struct Data *DataPtr;

void applyThreadFence() {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  // CIR-LABEL: @applyThreadFence
  // CIR:   cir.atomic.fence syncscope(system) seq_cst
  // CIR:   cir.return

  // LLVM-LABEL: @applyThreadFence
  // LLVM:    fence seq_cst
  // LLVM:    ret void

  // OGCG-LABEL: @applyThreadFence
  // OGCG:    fence seq_cst
  // OGCG:    ret void
}

void applySignalFence() {
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  // CIR-LABEL: @applySignalFence
  // CIR:    cir.atomic.fence syncscope(single_thread) seq_cst
  // CIR:    cir.return

  // LLVM-LABEL: @applySignalFence
  // LLVM:    fence syncscope("singlethread") seq_cst
  // LLVM:    ret void

  // OGCG-LABEL: @applySignalFence
  // OGCG:    fence syncscope("singlethread") seq_cst
  // OGCG:    ret void
}

void modifyWithThreadFence(DataPtr d) {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  d->value = 42;
  // CIR-LABEL: @modifyWithThreadFence
  // CIR:    %[[DATA:.*]] = cir.alloca !cir.ptr<!rec_Data>, !cir.ptr<!cir.ptr<!rec_Data>>, ["d", init] {alignment = 8 : i64}
  // CIR:    cir.atomic.fence syncscope(system) seq_cst
  // CIR:    %[[VAL_42:.*]] = cir.const #cir.int<42> : !s32i
  // CIR:    %[[LOAD_DATA:.*]] = cir.load{{.*}} %[[DATA]] : !cir.ptr<!cir.ptr<!rec_Data>>, !cir.ptr<!rec_Data>
  // CIR:    %[[DATA_VALUE:.*]] = cir.get_member %[[LOAD_DATA]][0] {name = "value"} : !cir.ptr<!rec_Data> -> !cir.ptr<!s32i>
  // CIR:    cir.store{{.*}} %[[VAL_42]], %[[DATA_VALUE]] : !s32i, !cir.ptr<!s32i>
  // CIR:    cir.return

  // LLVM-LABEL: @modifyWithThreadFence
  // LLVM:    %[[DATA:.*]] = alloca ptr, i64 1, align 8
  // LLVM:    fence seq_cst
  // LLVM:    %[[DATA_PTR:.*]] = load ptr, ptr %[[DATA]], align 8
  // LLVM:    %[[DATA_VALUE:.*]] = getelementptr %struct.Data, ptr %[[DATA_PTR]], i32 0, i32 0
  // LLVM:    store i32 42, ptr %[[DATA_VALUE]], align 8
  // LLVM:    ret void

  // OGCG-LABEL: @modifyWithThreadFence
  // OGCG:    %[[DATA:.*]] = alloca ptr, align 8
  // OGCG:    fence seq_cst
  // OGCG:    %[[DATA_PTR:.*]] = load ptr, ptr %[[DATA]], align 8
  // OGCG:    %[[DATA_VALUE:.*]] = getelementptr inbounds nuw %struct.Data, ptr %[[DATA_PTR]], i32 0, i32 0
  // OGCG:    store i32 42, ptr %[[DATA_VALUE]], align 8
  // OGCG:    ret void
}

void modifyWithSignalFence(DataPtr d) {
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  d->value = 24;
  // CIR-LABEL: @modifyWithSignalFence
  // CIR:    %[[DATA:.*]] = cir.alloca !cir.ptr<!rec_Data>, !cir.ptr<!cir.ptr<!rec_Data>>, ["d", init] {alignment = 8 : i64}
  // CIR:    cir.atomic.fence syncscope(single_thread) seq_cst
  // CIR:    %[[VAL_42:.*]] = cir.const #cir.int<24> : !s32i
  // CIR:    %[[LOAD_DATA:.*]] = cir.load{{.*}} %[[DATA]] : !cir.ptr<!cir.ptr<!rec_Data>>, !cir.ptr<!rec_Data>
  // CIR:    %[[DATA_VALUE:.*]] = cir.get_member %[[LOAD_DATA]][0] {name = "value"} : !cir.ptr<!rec_Data> -> !cir.ptr<!s32i>
  // CIR:    cir.store{{.*}} %[[VAL_42]], %[[DATA_VALUE]] : !s32i, !cir.ptr<!s32i>
  // CIR:    cir.return

  // LLVM-LABEL: @modifyWithSignalFence
  // LLVM:    %[[DATA:.*]] = alloca ptr, i64 1, align 8
  // LLVM:    fence syncscope("singlethread") seq_cst
  // LLVM:    %[[DATA_PTR:.*]] = load ptr, ptr %[[DATA]], align 8
  // LLVM:    %[[DATA_VALUE:.*]] = getelementptr %struct.Data, ptr %[[DATA_PTR]], i32 0, i32 0
  // LLVM:    store i32 24, ptr %[[DATA_VALUE]], align 8
  // LLVM:    ret void

  // OGCG-LABEL: @modifyWithSignalFence
  // OGCG:    %[[DATA:.*]] = alloca ptr, align 8
  // OGCG:    fence syncscope("singlethread") seq_cst
  // OGCG:    %[[DATA_PTR:.*]] = load ptr, ptr %[[DATA]], align 8
  // OGCG:    %[[DATA_VALUE:.*]] = getelementptr inbounds nuw %struct.Data, ptr %[[DATA_PTR]], i32 0, i32 0
  // OGCG:    store i32 24, ptr %[[DATA_VALUE]], align 8
  // OGCG:    ret void
}

void loadWithThreadFence(DataPtr d) {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  __atomic_load_n(&d->ptr, __ATOMIC_SEQ_CST);
  // CIR-LABEL: @loadWithThreadFence
  // CIR:    %[[DATA:.*]] = cir.alloca !cir.ptr<!rec_Data>, !cir.ptr<!cir.ptr<!rec_Data>>, ["d", init] {alignment = 8 : i64}
  // CIR:    %[[ATOMIC_TEMP:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["atomic-temp"] {alignment = 8 : i64}
  // CIR:    cir.atomic.fence syncscope(system) seq_cst
  // CIR:    %[[LOAD_DATA:.*]] = cir.load{{.*}} %[[DATA]] : !cir.ptr<!cir.ptr<!rec_Data>>, !cir.ptr<!rec_Data>
  // CIR:    %[[DATA_VALUE:.*]] = cir.get_member %[[LOAD_DATA]][1] {name = "ptr"} : !cir.ptr<!rec_Data> -> !cir.ptr<!cir.ptr<!void>>
  // CIR:    %[[CASTED_DATA_VALUE:.*]] = cir.cast bitcast %[[DATA_VALUE]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!u64i>
  // CIR:    %[[CASTED_ATOMIC_TEMP:.*]] = cir.cast bitcast %[[ATOMIC_TEMP]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!u64i>
  // CIR:    %[[ATOMIC_LOAD:.*]] = cir.load{{.*}} atomic(seq_cst) %[[CASTED_DATA_VALUE]] : !cir.ptr<!u64i>, !u64i
  // CIR:    cir.store{{.*}} %[[ATOMIC_LOAD]], %[[CASTED_ATOMIC_TEMP]] : !u64i, !cir.ptr<!u64i>
  // CIR:    %[[DOUBLE_CASTED_ATOMIC_TEMP:.*]] = cir.cast bitcast %[[CASTED_ATOMIC_TEMP]] : !cir.ptr<!u64i> -> !cir.ptr<!cir.ptr<!void>>
  // CIR:    %[[ATOMIC_LOAD_PTR:.*]] = cir.load{{.*}} %[[DOUBLE_CASTED_ATOMIC_TEMP]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR:    cir.return

  // LLVM-LABEL: @loadWithThreadFence
  // LLVM:    %[[DATA:.*]] = alloca ptr, i64 1, align 8
  // LLVM:    %[[DATA_TEMP:.*]] = alloca ptr, i64 1, align 8
  // LLVM:    fence seq_cst
  // LLVM:    %[[DATA_PTR:.*]] = load ptr, ptr %[[DATA]], align 8
  // LLVM:    %[[DATA_VALUE:.*]] = getelementptr %struct.Data, ptr %[[DATA_PTR]], i32 0, i32 1
  // LLVM:    %[[ATOMIC_LOAD:.*]] = load atomic i64, ptr %[[DATA_VALUE]] seq_cst, align 8
  // LLVM:    store i64 %[[ATOMIC_LOAD]], ptr %[[DATA_TEMP]], align 8
  // LLVM:    %[[DATA_TEMP_LOAD:.*]] = load ptr, ptr %[[DATA_TEMP]], align 8
  // LLVM:    ret void

  // OGCG-LABEL: @loadWithThreadFence
  // OGCG:    %[[DATA:.*]] = alloca ptr, align 8
  // OGCG:    %[[DATA_TEMP:.*]] = alloca ptr, align 8
  // OGCG:    fence seq_cst
  // OGCG:    %[[DATA_PTR:.*]] = load ptr, ptr %[[DATA]], align 8
  // OGCG:    %[[DATA_VALUE:.*]] = getelementptr inbounds nuw %struct.Data, ptr %[[DATA_PTR]], i32 0, i32 1
  // OGCG:    %[[ATOMIC_LOAD:.*]] = load atomic i64, ptr %[[DATA_VALUE]] seq_cst, align 8
  // OGCG:    store i64 %[[ATOMIC_LOAD]], ptr %[[DATA_TEMP]], align 8
  // OGCG:    %[[DATA_TEMP_LOAD:.*]] = load ptr, ptr %[[DATA_TEMP]], align 8
  // OGCG:    ret void
}

void loadWithSignalFence(DataPtr d) {
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  __atomic_load_n(&d->ptr, __ATOMIC_SEQ_CST);
  // CIR-LABEL: @loadWithSignalFence
  // CIR:    %[[DATA:.*]] = cir.alloca !cir.ptr<!rec_Data>, !cir.ptr<!cir.ptr<!rec_Data>>, ["d", init] {alignment = 8 : i64}
  // CIR:    %[[ATOMIC_TEMP:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["atomic-temp"] {alignment = 8 : i64}
  // CIR:    cir.atomic.fence syncscope(single_thread) seq_cst
  // CIR:    %[[LOAD_DATA:.*]] = cir.load{{.*}} %[[DATA]] : !cir.ptr<!cir.ptr<!rec_Data>>, !cir.ptr<!rec_Data>
  // CIR:    %[[DATA_PTR:.*]] = cir.get_member %[[LOAD_DATA]][1] {name = "ptr"} : !cir.ptr<!rec_Data> -> !cir.ptr<!cir.ptr<!void>>
  // CIR:    %[[CASTED_DATA_PTR:.*]] = cir.cast bitcast %[[DATA_PTR]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!u64i>
  // CIR:    %[[CASTED_ATOMIC_TEMP:.*]] = cir.cast bitcast %[[ATOMIC_TEMP]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!u64i>
  // CIR:    %[[ATOMIC_LOAD:.*]] = cir.load{{.*}} atomic(seq_cst) %[[CASTED_DATA_PTR]] : !cir.ptr<!u64i>, !u64i
  // CIR:    cir.store{{.*}} %[[ATOMIC_LOAD]], %[[CASTED_ATOMIC_TEMP]] : !u64i, !cir.ptr<!u64i>
  // CIR:    %[[DOUBLE_CASTED_ATOMIC_TEMP:.*]] = cir.cast bitcast %[[CASTED_ATOMIC_TEMP]] : !cir.ptr<!u64i> -> !cir.ptr<!cir.ptr<!void>>
  // CIR:    %[[LOAD_ATOMIC_TEMP:.*]] = cir.load{{.*}} %[[DOUBLE_CASTED_ATOMIC_TEMP]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR:    cir.return

  // LLVM-LABEL: @loadWithSignalFence
  // LLVM:    %[[DATA:.*]] = alloca ptr, i64 1, align 8
  // LLVM:    %[[DATA_TEMP:.*]] = alloca ptr, i64 1, align 8
  // LLVM:    fence syncscope("singlethread") seq_cst
  // LLVM:    %[[DATA_PTR:.*]] = load ptr, ptr %[[DATA]], align 8
  // LLVM:    %[[DATA_VALUE:.*]] = getelementptr %struct.Data, ptr %[[DATA_PTR]], i32 0, i32 1
  // LLVM:    %[[ATOMIC_LOAD:.*]] = load atomic i64, ptr %[[DATA_VALUE]] seq_cst, align 8
  // LLVM:    store i64 %[[ATOMIC_LOAD]], ptr %[[DATA_TEMP]], align 8
  // LLVM:    %[[DATA_TEMP_LOAD]] = load ptr, ptr %[[DATA_TEMP]], align 8
  // LLVM:    ret void

  // OGCG-LABEL: @loadWithSignalFence
  // OGCG:    %[[DATA:.*]] = alloca ptr, align 8
  // OGCG:    %[[DATA_TEMP:.*]] = alloca ptr, align 8
  // OGCG:    fence syncscope("singlethread") seq_cst
  // OGCG:    %[[DATA_PTR:.*]] = load ptr, ptr %[[DATA]], align 8
  // OGCG:    %[[DATA_VALUE:.*]] = getelementptr inbounds nuw %struct.Data, ptr %[[DATA_PTR]], i32 0, i32 1
  // OGCG:    %[[ATOMIC_LOAD:.*]] = load atomic i64, ptr %[[DATA_VALUE]] seq_cst, align 8
  // OGCG:    store i64 %[[ATOMIC_LOAD]], ptr %[[DATA_TEMP]], align 8
  // OGCG:    %[[DATA_TEMP_LOAD]] = load ptr, ptr %[[DATA_TEMP]], align 8
  // OGCG:    ret void
}
