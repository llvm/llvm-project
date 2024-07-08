// RUNAA: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --input-file=%t.cir -check-prefix=CIR %s

char buffer[32] = "This is a largely unused buffer";

// __builtin___clear_cache always maps to @llvm.clear_cache, but what
// each back-end produces is different, and this is tested in LLVM

// CIR-LABEL: main
// CIR:  %[[VAL_1:.*]] = cir.get_global @buffer : !cir.ptr<!cir.array<!s8i x 32>>
// CIR:  %[[VAL_2:.*]] = cir.cast(array_to_ptrdecay, %[[VAL_1]] : !cir.ptr<!cir.array<!s8i x 32>>), !cir.ptr<!s8i>
// CIR:  %[[VAL_3:.*]] = cir.cast(bitcast, %[[VAL_2]] : !cir.ptr<!s8i>), !cir.ptr<!void>
// CIR:  %[[VAL_4:.*]] = cir.get_global @buffer : !cir.ptr<!cir.array<!s8i x 32>>
// CIR:  %[[VAL_5:.*]] = cir.cast(array_to_ptrdecay, %[[VAL_4]] : !cir.ptr<!cir.array<!s8i x 32>>), !cir.ptr<!s8i>
// CIR:  %[[VAL_6:.*]] = cir.const #cir.int<32> : !s32i
// CIR:  %[[VAL_7:.*]] = cir.ptr_stride(%[[VAL_5]] : !cir.ptr<!s8i>, %[[VAL_6]] : !s32i), !cir.ptr<!s8i>
// CIR:  %[[VAL_8:.*]] = cir.cast(bitcast, %[[VAL_7]] : !cir.ptr<!s8i>), !cir.ptr<!void>
// CIR:  cir.clear_cache %[[VAL_3]] : !cir.ptr<!void>, %[[VAL_8]],

int main(void) {
  __builtin___clear_cache(buffer, buffer+32);
  return 0;
}
