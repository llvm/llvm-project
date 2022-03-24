// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s

func @_QPsb1(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr"}) {
  %c1_i64 = arith.constant 1 : i64
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsbEi"}
  omp.parallel  {
    %1 = fir.alloca i32 {adapt.valuebyref, pinned}
    %2 = fir.load %arg0 : !fir.ref<i32>
    omp.wsloop nowait
    for (%arg2) : i32 = (%c1_i32) to (%2) inclusive step (%c1_i32)  {
      fir.store %arg2 to %1 : !fir.ref<i32>
      %3 = fir.load %1 : !fir.ref<i32>
      %4 = fir.convert %3 : (i32) -> i64
      %5 = arith.subi %4, %c1_i64 : i64
      %6 = fir.coordinate_of %arg1, %5 : (!fir.ref<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
      fir.store %3 to %6 : !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL:  _QPsb1
// CHECK-SAME: %[[N_REF:.*]]: !llvm.ptr<i32> {fir.bindc_name = "n"}, %[[ARR_REF:.*]]: !llvm.ptr<i32> {fir.bindc_name = "arr"}) {
// CHECK:    %[[ONE_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[ONE_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: omp.parallel   {
// CHECK:      %[[ONE_3:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:      %[[I_VAR:.*]] = llvm.alloca %[[ONE_3]] x i32 {adapt.valuebyref, in_type = i32, operand_segment_sizes = dense<0> : vector<2xi32>, pinned} : (i64) -> !llvm.ptr<i32>
// CHECK:      %[[N:.*]] = llvm.load %[[N_REF]] : !llvm.ptr<i32>
// CHECK: omp.wsloop nowait
// CHECK-SAME: for (%[[I:.*]]) : i32 = (%[[ONE_2]]) to (%[[N]]) inclusive step (%[[ONE_2]]) {
// CHECK:   llvm.store %[[I]], %[[I_VAR]] : !llvm.ptr<i32>
// CHECK:   %[[I1:.*]] = llvm.load %[[I_VAR]] : !llvm.ptr<i32>
// CHECK:   %[[I1_EXT:.*]] = llvm.sext %[[I1]] : i32 to i64
// CHECK:   %[[I_CSTYLE:.*]] = llvm.sub %[[I1_EXT]], %[[ONE_1]]  : i64
// CHECK:   %[[ARR_I_REF:.*]] = llvm.getelementptr %[[ARR_REF]][%[[I_CSTYLE]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK:   llvm.store %[[I1]], %[[ARR_I_REF]] : !llvm.ptr<i32>
// CHECK: omp.yield
// CHECK: }
// CHECK: omp.terminator
// CHECK: }
// CHECK: llvm.return
// CHECK: }

// -----

func @_QPsb2(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  omp.parallel  {
    omp.master  {
      %0 = fir.load %arg1 : !fir.ref<i32>
      fir.store %0 to %arg0 : !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: _QPsb2
// CHECK-SAME: %[[X_REF:.*]]: !llvm.ptr<i32> {fir.bindc_name = "x"}, %[[N_REF:.*]]: !llvm.ptr<i32> {fir.bindc_name = "n"}) {
// CHECK: omp.parallel   {
// CHECK:   omp.master {
// CHECK:     %[[N:.*]] = llvm.load %[[N_REF]] : !llvm.ptr<i32>
// CHECK:     llvm.store %[[N]], %[[X_REF]] : !llvm.ptr<i32>
// CHECK:     omp.terminator
// CHECK:   }
// CHECK:   omp.terminator
// CHECK: }
// CHECK: llvm.return
// CHECK: }
