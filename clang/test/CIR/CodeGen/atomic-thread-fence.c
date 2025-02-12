// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s


struct Data {
  int value;
  void *ptr;
};

typedef struct Data *DataPtr;

void applyThreadFence() {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

// CIR-LABEL: @applyThreadFence
// CIR:   cir.atomic.fence system seq_cst
// CIR:   cir.return

// LLVM-LABEL: @applyThreadFence
// LLVM:    fence seq_cst
// LLVM:    ret void

void applySignalFence() {
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
}
// CIR-LABEL: @applySignalFence
// CIR:    cir.atomic.fence single_thread seq_cst
// CIR:    cir.return

// LLVM-LABEL: @applySignalFence
// LLVM:    fence syncscope("singlethread") seq_cst
// LLVM:    ret void

void modifyWithThreadFence(DataPtr d) {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  d->value = 42;
}
// CIR-LABEL: @modifyWithThreadFence
// CIR:    %[[DATA:.*]] = cir.alloca !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>, ["d", init] {alignment = 8 : i64}
// CIR:    cir.atomic.fence system seq_cst
// CIR:    %[[VAL_42:.*]] = cir.const #cir.int<42> : !s32i
// CIR:    %[[LOAD_DATA:.*]] = cir.load %[[DATA]] : !cir.ptr<!cir.ptr<!ty_Data>>, !cir.ptr<!ty_Data>
// CIR:    %[[DATA_VALUE:.*]] = cir.get_member %[[LOAD_DATA]][0] {name = "value"} : !cir.ptr<!ty_Data> -> !cir.ptr<!s32i>
// CIR:    cir.store %[[VAL_42]], %[[DATA_VALUE]] : !s32i, !cir.ptr<!s32i>
// CIR:    cir.return

// LLVM-LABEL: @modifyWithThreadFence
// LLVM:    %[[DATA:.*]] = alloca ptr, i64 1, align 8
// LLVM:    fence seq_cst
// LLVM:    %[[DATA_PTR:.*]] = load ptr, ptr %[[DATA]], align 8
// LLVM:    %[[DATA_VALUE:.*]] = getelementptr %struct.Data, ptr %[[DATA_PTR]], i32 0, i32 0
// LLVM:    store i32 42, ptr %[[DATA_VALUE]], align 4
// LLVM:    ret void

void modifyWithSignalFence(DataPtr d) {
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  d->value = 24;
}
// CIR-LABEL: @modifyWithSignalFence
// CIR:    %[[DATA:.*]] = cir.alloca !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>, ["d", init] {alignment = 8 : i64}
// CIR:    cir.atomic.fence single_thread seq_cst
// CIR:    %[[VAL_42:.*]] = cir.const #cir.int<24> : !s32i
// CIR:    %[[LOAD_DATA:.*]] = cir.load %[[DATA]] : !cir.ptr<!cir.ptr<!ty_Data>>, !cir.ptr<!ty_Data>
// CIR:    %[[DATA_VALUE:.*]] = cir.get_member %[[LOAD_DATA]][0] {name = "value"} : !cir.ptr<!ty_Data> -> !cir.ptr<!s32i>
// CIR:    cir.store %[[VAL_42]], %[[DATA_VALUE]] : !s32i, !cir.ptr<!s32i>
// CIR:    cir.return

// LLVM-LABEL: @modifyWithSignalFence
// LLVM:    %[[DATA:.*]] = alloca ptr, i64 1, align 8
// LLVM:    fence syncscope("singlethread") seq_cst
// LLVM:    %[[DATA_PTR:.*]] = load ptr, ptr %[[DATA]], align 8
// LLVM:    %[[DATA_VALUE:.*]] = getelementptr %struct.Data, ptr %[[DATA_PTR]], i32 0, i32 0
// LLVM:    store i32 24, ptr %[[DATA_VALUE]], align 4
// LLVM:    ret void

void loadWithThreadFence(DataPtr d) {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  __atomic_load_n(&d->ptr, __ATOMIC_SEQ_CST);
}
// CIR-LABEL: @loadWithThreadFence
// CIR:    %[[DATA:.*]] = cir.alloca !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>, ["d", init] {alignment = 8 : i64}
// CIR:    %[[ATOMIC_TEMP:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["atomic-temp"] {alignment = 8 : i64}
// CIR:    cir.atomic.fence system seq_cst
// CIR:    %[[LOAD_DATA:.*]] = cir.load %[[DATA]] : !cir.ptr<!cir.ptr<!ty_Data>>, !cir.ptr<!ty_Data>
// CIR:    %[[DATA_VALUE:.*]] = cir.get_member %[[LOAD_DATA]][1] {name = "ptr"} : !cir.ptr<!ty_Data> -> !cir.ptr<!cir.ptr<!void>>
// CIR:    %[[CASTED_DATA_VALUE:.*]] = cir.cast(bitcast, %[[DATA_VALUE]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!u64i>
// CIR:    %[[ATOMIC_LOAD:.*]] = cir.load atomic(seq_cst) %[[CASTED_DATA_VALUE]] : !cir.ptr<!u64i>, !u64i
// CIR:    %[[CASTED_ATOMIC_TEMP:.*]] = cir.cast(bitcast, %[[ATOMIC_TEMP]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!u64i>
// CIR:    cir.store %[[ATOMIC_LOAD]], %[[CASTED_ATOMIC_TEMP]] : !u64i, !cir.ptr<!u64i>
// CIR:    %[[ATOMIC_LOAD_PTR:.*]] = cir.load %[[ATOMIC_TEMP]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
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

void loadWithSignalFence(DataPtr d) {
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  __atomic_load_n(&d->ptr, __ATOMIC_SEQ_CST);
}
// CIR-LABEL: @loadWithSignalFence
// CIR:    %[[DATA:.*]] = cir.alloca !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>, ["d", init] {alignment = 8 : i64}
// CIR:    %[[ATOMIC_TEMP:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["atomic-temp"] {alignment = 8 : i64}
// CIR:    cir.atomic.fence single_thread seq_cst
// CIR:    %[[LOAD_DATA:.*]] = cir.load %[[DATA]] : !cir.ptr<!cir.ptr<!ty_Data>>, !cir.ptr<!ty_Data>
// CIR:    %[[DATA_PTR:.*]] = cir.get_member %[[LOAD_DATA]][1] {name = "ptr"} : !cir.ptr<!ty_Data> -> !cir.ptr<!cir.ptr<!void>>
// CIR:    %[[CASTED_DATA_PTR:.*]] = cir.cast(bitcast, %[[DATA_PTR]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!u64i>
// CIR:    %[[ATOMIC_LOAD:.*]] = cir.load atomic(seq_cst) %[[CASTED_DATA_PTR]] : !cir.ptr<!u64i>, !u64i
// CIR:    %[[CASTED_ATOMIC_TEMP:.*]] = cir.cast(bitcast, %[[ATOMIC_TEMP]] : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!u64i>
// CIR:    cir.store %[[ATOMIC_LOAD]], %[[CASTED_ATOMIC_TEMP]] : !u64i, !cir.ptr<!u64i>
// CIR:    %[[LOAD_ATOMIC_TEMP:.*]] = cir.load %[[ATOMIC_TEMP]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
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
