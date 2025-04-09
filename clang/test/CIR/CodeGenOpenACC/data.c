// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -fopenacc -emit-llvm -fclangir %s -o - | FileCheck %s -check-prefix=LLVM

void acc_data(void) {
  // CIR: cir.func @acc_data() {
  // LLVM: define void @acc_data() {

#pragma acc data default(none)
  {
    int i = 0;
    ++i;
  }
  // CIR-NEXT: acc.data {
  // CIR-NEXT: cir.alloca
  // CIR-NEXT: cir.const
  // CIR-NEXT: cir.store
  // CIR-NEXT: cir.load
  // CIR-NEXT: cir.unary
  // CIR-NEXT: cir.store
  // CIR-NEXT: acc.terminator
  // CIR-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}
  //
  // LLVM: call void @__tgt_target_data_begin_mapper
  // LLVM-NEXT: br label %[[ACC_DATA:.+]]
  // LLVM: [[ACC_DATA]]:
  // LLVM-NEXT: store i32 0
  // LLVM-NEXT: load i32
  // LLVM-NEXT: add nsw i32 %{{.*}}, 1
  // LLVM-NEXT: store i32
  // LLVM-NEXT: br label %[[ACC_DATA_END:.+]]
  //
  // LLVM: [[ACC_DATA_END]]:
  // LLVM: call void @__tgt_target_data_end_mapper

#pragma acc data default(present)
  {
    int i = 0;
    ++i;
  }
  // CIR-NEXT: acc.data {
  // CIR-NEXT: cir.alloca
  // CIR-NEXT: cir.const
  // CIR-NEXT: cir.store
  // CIR-NEXT: cir.load
  // CIR-NEXT: cir.unary
  // CIR-NEXT: cir.store
  // CIR-NEXT: acc.terminator
  // CIR-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}

  // LLVM: call void @__tgt_target_data_begin_mapper
  // LLVM-NEXT: br label %[[ACC_DATA:.+]]
  // LLVM: [[ACC_DATA]]:
  // LLVM-NEXT: store i32 0
  // LLVM-NEXT: load i32
  // LLVM-NEXT: add nsw i32 %{{.*}}, 1
  // LLVM-NEXT: store i32
  // LLVM-NEXT: br label %[[ACC_DATA_END:.+]]
  //
  // LLVM: [[ACC_DATA_END]]:
  // LLVM: call void @__tgt_target_data_end_mapper

  // CIR-NEXT: cir.return
  // LLVM: ret void
}
