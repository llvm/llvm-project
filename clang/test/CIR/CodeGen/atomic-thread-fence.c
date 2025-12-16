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

void const_atomic_thread_fence() {
  __atomic_thread_fence(__ATOMIC_RELAXED);
  __atomic_thread_fence(__ATOMIC_CONSUME);
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  __atomic_thread_fence(__ATOMIC_RELEASE);
  __atomic_thread_fence(__ATOMIC_ACQ_REL);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  // CIR-LABEL: const_atomic_thread_fence
  // CIR: cir.atomic.fence syncscope(system) acquire
  // CIR: cir.atomic.fence syncscope(system) acquire
  // CIR: cir.atomic.fence syncscope(system) release
  // CIR: cir.atomic.fence syncscope(system) acq_rel
  // CIR: cir.atomic.fence syncscope(system) seq_cst

  // LLVM-LABEL: const_atomic_thread_fence
  // LLVM: fence acquire
  // LLVM: fence acquire
  // LLVM: fence release
  // LLVM: fence acq_rel
  // LLVM: fence seq_cst

  // OGCG-LABEL: const_atomic_thread_fence
  // OGCG: fence acquire
  // OGCG: fence acquire
  // OGCG: fence release
  // OGCG: fence acq_rel
  // OGCG: fence seq_cst
}

void const_c11_atomic_thread_fence() {
  __c11_atomic_thread_fence(__ATOMIC_RELAXED);
  __c11_atomic_thread_fence(__ATOMIC_CONSUME);
  __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);
  __c11_atomic_thread_fence(__ATOMIC_RELEASE);
  __c11_atomic_thread_fence(__ATOMIC_ACQ_REL);
  __c11_atomic_thread_fence(__ATOMIC_SEQ_CST);
  // CIR-LABEL: const_c11_atomic_thread_fence
  // CIR: cir.atomic.fence syncscope(system) acquire
  // CIR: cir.atomic.fence syncscope(system) acquire
  // CIR: cir.atomic.fence syncscope(system) release
  // CIR: cir.atomic.fence syncscope(system) acq_rel
  // CIR: cir.atomic.fence syncscope(system) seq_cst

  // LLVM-LABEL: const_c11_atomic_thread_fence
  // LLVM: fence acquire
  // LLVM: fence acquire
  // LLVM: fence release
  // LLVM: fence acq_rel
  // LLVM: fence seq_cst

  // OGCG-LABEL: const_c11_atomic_thread_fence
  // OGCG: fence acquire
  // OGCG: fence acquire
  // OGCG: fence release
  // OGCG: fence acq_rel
  // OGCG: fence seq_cst
}

void const_atomic_signal_fence() {
  __atomic_signal_fence(__ATOMIC_RELAXED);
  __atomic_signal_fence(__ATOMIC_CONSUME);
  __atomic_signal_fence(__ATOMIC_ACQUIRE);
  __atomic_signal_fence(__ATOMIC_RELEASE);
  __atomic_signal_fence(__ATOMIC_ACQ_REL);
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  // CIR-LABEL: const_atomic_signal_fence
  // CIR: cir.atomic.fence syncscope(single_thread) acquire
  // CIR: cir.atomic.fence syncscope(single_thread) acquire
  // CIR: cir.atomic.fence syncscope(single_thread) release
  // CIR: cir.atomic.fence syncscope(single_thread) acq_rel
  // CIR: cir.atomic.fence syncscope(single_thread) seq_cst

  // LLVM-LABEL: const_atomic_signal_fence
  // LLVM: fence syncscope("singlethread") acquire
  // LLVM: fence syncscope("singlethread") acquire
  // LLVM: fence syncscope("singlethread") release
  // LLVM: fence syncscope("singlethread") acq_rel
  // LLVM: fence syncscope("singlethread") seq_cst

  // OGCG--LABEL: const_atomic_signal_fence
  // OGCG: fence syncscope("singlethread") acquire
  // OGCG: fence syncscope("singlethread") acquire
  // OGCG: fence syncscope("singlethread") release
  // OGCG: fence syncscope("singlethread") acq_rel
  // OGCG: fence syncscope("singlethread") seq_cst
}

void const_c11_atomic_signal_fence() {
  __c11_atomic_signal_fence(__ATOMIC_RELAXED);
  __c11_atomic_signal_fence(__ATOMIC_CONSUME);
  __c11_atomic_signal_fence(__ATOMIC_ACQUIRE);
  __c11_atomic_signal_fence(__ATOMIC_RELEASE);
  __c11_atomic_signal_fence(__ATOMIC_ACQ_REL);
  __c11_atomic_signal_fence(__ATOMIC_SEQ_CST);
  // CIR-LABEL: const_c11_atomic_signal_fence
  // CIR: cir.atomic.fence syncscope(single_thread) acquire
  // CIR: cir.atomic.fence syncscope(single_thread) acquire
  // CIR: cir.atomic.fence syncscope(single_thread) release
  // CIR: cir.atomic.fence syncscope(single_thread) acq_rel
  // CIR: cir.atomic.fence syncscope(single_thread) seq_cst

  // LLVM-LABEL: const_c11_atomic_signal_fence
  // LLVM: fence syncscope("singlethread") acquire
  // LLVM: fence syncscope("singlethread") acquire
  // LLVM: fence syncscope("singlethread") release
  // LLVM: fence syncscope("singlethread") acq_rel
  // LLVM: fence syncscope("singlethread") seq_cst

  // OGCG-LABEL: const_c11_atomic_signal_fence
  // OGCG: fence syncscope("singlethread") acquire
  // OGCG: fence syncscope("singlethread") acquire
  // OGCG: fence syncscope("singlethread") release
  // OGCG: fence syncscope("singlethread") acq_rel
  // OGCG: fence syncscope("singlethread") seq_cst
}

void variable_atomic_thread_fences(int memorder) {
  __atomic_thread_fence(memorder);
  // CIR-LABEL: variable_atomic_thread_fences
  // CIR:  cir.switch
  // CIR:    cir.case(default, []) {
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(system) acquire
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<3> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(system) release
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<4> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(system) acq_rel
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<5> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(system)
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.yield
  // CIR:  }

  // LLVM-LABEL: variable_atomic_thread_fences
  // LLVM:   %[[ORDER:.+]] = load i32, ptr %[[PTR:.+]], align 4
  // LLVM:   br label %[[SWITCH_BLK:.+]]
  // LLVM: [[SWITCH_BLK]]:
  // LLVM:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // LLVM:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // LLVM:     i32 2, label %[[ACQUIRE_BLK]]
  // LLVM:     i32 3, label %[[RELEASE_BLK:.+]]
  // LLVM:     i32 4, label %[[ACQ_REL_BLK:.+]]
  // LLVM:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // LLVM:   ]
  // LLVM: [[DEFAULT_BLK]]:
  // LLVM:   br label %{{.+}}
  // LLVM: [[ACQUIRE_BLK]]:
  // LLVM:   fence acquire
  // LLVM:   br label %{{.+}}
  // LLVM: [[RELEASE_BLK]]:
  // LLVM:   fence release
  // LLVM:   br label %{{.+}}
  // LLVM: [[ACQ_REL_BLK]]:
  // LLVM:   fence acq_rel
  // LLVM:   br label %{{.+}}
  // LLVM: [[SEQ_CST_BLK]]:
  // LLVM:   fence seq_cst
  // LLVM:   br label %{{.+}}

  // OGCG-LABEL: variable_atomic_thread_fences
  // OGCG:   %[[ORDER:.+]] = load i32, ptr %[[PTR:.+]], align 4
  // OGCG:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // OGCG:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // OGCG:     i32 2, label %[[ACQUIRE_BLK]]
  // OGCG:     i32 3, label %[[RELEASE_BLK:.+]]
  // OGCG:     i32 4, label %[[ACQ_REL_BLK:.+]]
  // OGCG:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // OGCG:   ]
  // OGCG: [[ACQUIRE_BLK]]:
  // OGCG:   fence acquire
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[RELEASE_BLK]]:
  // OGCG:   fence release
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[ACQ_REL_BLK]]:
  // OGCG:   fence acq_rel
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[SEQ_CST_BLK]]:
  // OGCG:   fence seq_cst
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[DEFAULT_BLK]]:
  // OGCG:   ret void
}

void variable_c11_atomic_thread_fences(int memorder) {
  __c11_atomic_thread_fence(memorder);
  // CIR-LABEL: variable_c11_atomic_thread_fences
  // CIR:  cir.switch
  // CIR:    cir.case(default, []) {
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(system) acquire
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<3> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(system) release
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<4> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(system) acq_rel
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<5> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(system)
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.yield
  // CIR:  }

  // LLVM-LABEL: variable_c11_atomic_thread_fences
  // LLVM:   %[[ORDER:.+]] = load i32, ptr %[[PTR:.+]], align 4
  // LLVM:   br label %[[SWITCH_BLK:.+]]
  // LLVM: [[SWITCH_BLK]]:
  // LLVM:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // LLVM:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // LLVM:     i32 2, label %[[ACQUIRE_BLK]]
  // LLVM:     i32 3, label %[[RELEASE_BLK:.+]]
  // LLVM:     i32 4, label %[[ACQ_REL_BLK:.+]]
  // LLVM:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // LLVM:   ]
  // LLVM: [[DEFAULT_BLK]]:
  // LLVM:   br label %{{.+}}
  // LLVM: [[ACQUIRE_BLK]]:
  // LLVM:   fence acquire
  // LLVM:   br label %{{.+}}
  // LLVM: [[RELEASE_BLK]]:
  // LLVM:   fence release
  // LLVM:   br label %{{.+}}
  // LLVM: [[ACQ_REL_BLK]]:
  // LLVM:   fence acq_rel
  // LLVM:   br label %{{.+}}
  // LLVM: [[SEQ_CST_BLK]]:
  // LLVM:   fence seq_cst
  // LLVM:   br label %{{.+}}

  // OGCG-LABEL: variable_c11_atomic_thread_fences
  // OGCG:   %[[ORDER:.+]] = load i32, ptr %[[PTR:.+]], align 4
  // OGCG:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // OGCG:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // OGCG:     i32 2, label %[[ACQUIRE_BLK]]
  // OGCG:     i32 3, label %[[RELEASE_BLK:.+]]
  // OGCG:     i32 4, label %[[ACQ_REL_BLK:.+]]
  // OGCG:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // OGCG:   ]
  // OGCG: [[ACQUIRE_BLK]]:
  // OGCG:   fence acquire
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[RELEASE_BLK]]:
  // OGCG:   fence release
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[ACQ_REL_BLK]]:
  // OGCG:   fence acq_rel
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[SEQ_CST_BLK]]:
  // OGCG:   fence seq_cst
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[DEFAULT_BLK]]:
  // OGCG:   ret void
}

void variable_atomic_signal_fences(int memorder) {
  __atomic_signal_fence(memorder);
  // CIR-LABEL: variable_atomic_signal_fences
  // CIR:  cir.switch
  // CIR:    cir.case(default, []) {
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(single_thread) acquire
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<3> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(single_thread) release
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<4> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(single_thread) acq_rel
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<5> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(single_thread)
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.yield
  // CIR:  }

  // LLVM-LABEL: variable_atomic_signal_fences
  // LLVM:   %[[ORDER:.+]] = load i32, ptr %[[PTR:.+]], align 4
  // LLVM:   br label %[[SWITCH_BLK:.+]]
  // LLVM: [[SWITCH_BLK]]:
  // LLVM:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // LLVM:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // LLVM:     i32 2, label %[[ACQUIRE_BLK]]
  // LLVM:     i32 3, label %[[RELEASE_BLK:.+]]
  // LLVM:     i32 4, label %[[ACQ_REL_BLK:.+]]
  // LLVM:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // LLVM:   ]
  // LLVM: [[DEFAULT_BLK]]:
  // LLVM:   br label %{{.+}}
  // LLVM: [[ACQUIRE_BLK]]:
  // LLVM:   fence syncscope("singlethread") acquire
  // LLVM:   br label %{{.+}}
  // LLVM: [[RELEASE_BLK]]:
  // LLVM:   fence syncscope("singlethread") release
  // LLVM:   br label %{{.+}}
  // LLVM: [[ACQ_REL_BLK]]:
  // LLVM:   fence syncscope("singlethread") acq_rel
  // LLVM:   br label %{{.+}}
  // LLVM: [[SEQ_CST_BLK]]:
  // LLVM:   fence syncscope("singlethread") seq_cst
  // LLVM:   br label %{{.+}}

  // OGCG-LABEL: variable_atomic_signal_fences
  // OGCG:   %[[ORDER:.+]] = load i32, ptr %[[PTR:.+]], align 4
  // OGCG:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // OGCG:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // OGCG:     i32 2, label %[[ACQUIRE_BLK]]
  // OGCG:     i32 3, label %[[RELEASE_BLK:.+]]
  // OGCG:     i32 4, label %[[ACQ_REL_BLK:.+]]
  // OGCG:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // OGCG:   ]
  // OGCG: [[ACQUIRE_BLK]]:
  // OGCG:   fence syncscope("singlethread") acquire
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[RELEASE_BLK]]:
  // OGCG:   fence syncscope("singlethread") release
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[ACQ_REL_BLK]]:
  // OGCG:   fence syncscope("singlethread") acq_rel
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[SEQ_CST_BLK]]:
  // OGCG:   fence syncscope("singlethread") seq_cst
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[DEFAULT_BLK]]:
  // OGCG:   ret void
}

void variable_c11_atomic_signal_fences(int memorder) {
  __c11_atomic_signal_fence(memorder);
  // CIR-LABEL: variable_c11_atomic_signal_fences
  // CIR:  cir.switch
  // CIR:    cir.case(default, []) {
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<1> : !s32i, #cir.int<2> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(single_thread) acquire
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<3> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(single_thread) release
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<4> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(single_thread) acq_rel
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.case(anyof, [#cir.int<5> : !s32i]) {
  // CIR:      cir.atomic.fence syncscope(single_thread)
  // CIR:      cir.break
  // CIR:    }
  // CIR:    cir.yield
  // CIR:  }

  // LLVM-LABEL: variable_c11_atomic_signal_fences
  // LLVM:   %[[ORDER:.+]] = load i32, ptr %[[PTR:.+]], align 4
  // LLVM:   br label %[[SWITCH_BLK:.+]]
  // LLVM: [[SWITCH_BLK]]:
  // LLVM:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // LLVM:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // LLVM:     i32 2, label %[[ACQUIRE_BLK]]
  // LLVM:     i32 3, label %[[RELEASE_BLK:.+]]
  // LLVM:     i32 4, label %[[ACQ_REL_BLK:.+]]
  // LLVM:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // LLVM:   ]
  // LLVM: [[DEFAULT_BLK]]:
  // LLVM:   br label %{{.+}}
  // LLVM: [[ACQUIRE_BLK]]:
  // LLVM:   fence syncscope("singlethread") acquire
  // LLVM:   br label %{{.+}}
  // LLVM: [[RELEASE_BLK]]:
  // LLVM:   fence syncscope("singlethread") release
  // LLVM:   br label %{{.+}}
  // LLVM: [[ACQ_REL_BLK]]:
  // LLVM:   fence syncscope("singlethread") acq_rel
  // LLVM:   br label %{{.+}}
  // LLVM: [[SEQ_CST_BLK]]:
  // LLVM:   fence syncscope("singlethread") seq_cst
  // LLVM:   br label %{{.+}}

  // OGCG-LABEL: variable_c11_atomic_signal_fences
  // OGCG:   %[[ORDER:.+]] = load i32, ptr %[[PTR:.+]], align 4
  // OGCG:   switch i32 %[[ORDER]], label %[[DEFAULT_BLK:.+]] [
  // OGCG:     i32 1, label %[[ACQUIRE_BLK:.+]]
  // OGCG:     i32 2, label %[[ACQUIRE_BLK]]
  // OGCG:     i32 3, label %[[RELEASE_BLK:.+]]
  // OGCG:     i32 4, label %[[ACQ_REL_BLK:.+]]
  // OGCG:     i32 5, label %[[SEQ_CST_BLK:.+]]
  // OGCG:   ]
  // OGCG: [[ACQUIRE_BLK]]:
  // OGCG:   fence syncscope("singlethread") acquire
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[RELEASE_BLK]]:
  // OGCG:   fence syncscope("singlethread") release
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[ACQ_REL_BLK]]:
  // OGCG:   fence syncscope("singlethread") acq_rel
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[SEQ_CST_BLK]]:
  // OGCG:   fence syncscope("singlethread") seq_cst
  // OGCG:   br label %[[DEFAULT_BLK]]
  // OGCG: [[DEFAULT_BLK]]:
  // OGCG:   ret void
}
