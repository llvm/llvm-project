// RUN: fir-opt --split-input-file --cfg-conversion --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s

func.func @_QPsb1(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr"}) {
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
// CHECK:      %[[I_VAR:.*]] = llvm.alloca %[[ONE_3]] x i32 {adapt.valuebyref, in_type = i32, operandSegmentSizes = array<i32: 0, 0>, pinned} : (i64) -> !llvm.ptr<i32>
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

func.func @_QPsb2(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
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

// -----

func.func @_QPsb(%arr: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr"}) {
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsbEi"}
  omp.parallel   {
    %c1 = arith.constant 1 : i32
    %c50 = arith.constant 50 : i32
    omp.wsloop   for  (%indx) : i32 = (%c1) to (%c50) inclusive step (%c1) {
      %1 = fir.convert %indx : (i32) -> i64
      %c1_i64 = arith.constant 1 : i64
      %2 = arith.subi %1, %c1_i64 : i64
      %3 = fir.coordinate_of %arr, %2 : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
      fir.store %indx to %3 : !fir.ref<i32>
      omp.yield
    }
    omp.terminator
  }
  return
}

// Check only for the structure of the OpenMP portion and the feasibility of the conversion
// CHECK-LABEL: @_QPsb
// CHECK-SAME: %{{.*}}: !llvm.ptr<struct<({{.*}})>> {fir.bindc_name = "arr"}
// CHECK:    omp.parallel   {
// CHECK:      %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:      %[[C50:.*]] = llvm.mlir.constant(50 : i32) : i32
// CHECK:      omp.wsloop   for  (%[[INDX:.*]]) : i32 = (%[[C1]]) to (%[[C50]]) inclusive step (%[[C1]]) {
// CHECK:        llvm.store %[[INDX]], %{{.*}} : !llvm.ptr<i32>
// CHECK:        omp.yield
// CHECK:      omp.terminator
// CHECK:    llvm.return

// -----

func.func private @foo()
func.func private @bar()

func.func @sections_no_data() {
  omp.sections {
    omp.section {
      fir.call @foo() : () -> ()
      omp.terminator
    }
    omp.section {
      fir.call @bar() : () -> ()
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: llvm.func @sections_no_data
// CHECK: omp.sections {
// CHECK:   omp.section {
// CHECK:     llvm.call @foo() : () -> ()
// CHECK:     omp.terminator
// CHECK:   }
// CHECK:   omp.section {
// CHECK:     llvm.call @bar() : () -> ()
// CHECK:     omp.terminator
// CHECK:   }
// CHECK:   omp.terminator
// CHECK: }

// -----

func.func private @foo(!fir.ref<i32>)
func.func private @bar(!fir.ref<i32>, !fir.ref<i32>)

func.func @sections_data_without_clauses(%arg0: !fir.ref<i32> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "b"}) {
  omp.sections {
    omp.section {
      fir.call @foo(%arg0) : (!fir.ref<i32>) -> ()
      omp.terminator
    }
    omp.section {
      fir.call @bar(%arg0, %arg1) : (!fir.ref<i32>, !fir.ref<i32>) -> ()
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: llvm.func @sections_data_without_clauses
// CHECK-SAME:            (%[[ARG0:.+]]: !llvm.ptr<i32> {fir.bindc_name = "a"}, %[[ARG1:.+]]: !llvm.ptr<i32> {fir.bindc_name = "b"})
// CHECK: omp.sections {
// CHECK:   omp.section {
// CHECK:     llvm.call @foo(%arg0) : (!llvm.ptr<i32>) -> ()
// CHECK:     omp.terminator
// CHECK:   }
// CHECK:   omp.section {
// CHECK:     llvm.call @bar(%[[ARG0]], %[[ARG1]]) : (!llvm.ptr<i32>, !llvm.ptr<i32>) -> ()
// CHECK:     omp.terminator
// CHECK:   }
// CHECK:   omp.terminator
// CHECK: }

// -----

func.func @_QPsimd1(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr"}) {
  %c1_i64 = arith.constant 1 : i64
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsbEi"}
  omp.parallel  {
    %1 = fir.alloca i32 {adapt.valuebyref, pinned}
    %2 = fir.load %arg0 : !fir.ref<i32>
    omp.simdloop for (%arg2) : i32 = (%c1_i32) to (%2) step (%c1_i32)  {
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

// CHECK-LABEL:  _QPsimd1
// CHECK-SAME: %[[N_REF:.*]]: !llvm.ptr<i32> {fir.bindc_name = "n"}, %[[ARR_REF:.*]]: !llvm.ptr<i32> {fir.bindc_name = "arr"}) {
// CHECK:    %[[ONE_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[ONE_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: omp.parallel   {
// CHECK:      %[[ONE_3:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:      %[[I_VAR:.*]] = llvm.alloca %[[ONE_3]] x i32 {adapt.valuebyref, in_type = i32, operandSegmentSizes = array<i32: 0, 0>, pinned} : (i64) -> !llvm.ptr<i32>
// CHECK:      %[[N:.*]] = llvm.load %[[N_REF]] : !llvm.ptr<i32>
// CHECK: omp.simdloop
// CHECK-SAME: (%[[I:.*]]) : i32 = (%[[ONE_2]]) to (%[[N]]) step (%[[ONE_2]]) {
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

func.func @_QPomp_target_data() {
  %0 = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFomp_target_dataEa"}
  %1 = fir.alloca !fir.array<1024xi32> {bindc_name = "b", uniq_name = "_QFomp_target_dataEb"}
  %2 = fir.alloca !fir.array<1024xi32> {bindc_name = "c", uniq_name = "_QFomp_target_dataEc"}
  %3 = fir.alloca !fir.array<1024xi32> {bindc_name = "d", uniq_name = "_QFomp_target_dataEd"}
  omp.target_enter_data   map((to -> %0 : !fir.ref<!fir.array<1024xi32>>), (to -> %1 : !fir.ref<!fir.array<1024xi32>>), (always, alloc -> %2 : !fir.ref<!fir.array<1024xi32>>))
  omp.target_exit_data   map((from -> %0 : !fir.ref<!fir.array<1024xi32>>), (from -> %1 : !fir.ref<!fir.array<1024xi32>>), (release -> %2 : !fir.ref<!fir.array<1024xi32>>), (always, delete -> %3 : !fir.ref<!fir.array<1024xi32>>))
  return
}

// CHECK-LABEL:   llvm.func @_QPomp_target_data() {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<1024 x i32> {bindc_name = "a", in_type = !fir.array<1024xi32>, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_dataEa"} : (i64) -> !llvm.ptr<array<1024 x i32>>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.array<1024 x i32> {bindc_name = "b", in_type = !fir.array<1024xi32>, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_dataEb"} : (i64) -> !llvm.ptr<array<1024 x i32>>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_4]] x !llvm.array<1024 x i32> {bindc_name = "c", in_type = !fir.array<1024xi32>, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_dataEc"} : (i64) -> !llvm.ptr<array<1024 x i32>>
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_7:.*]] = llvm.alloca %[[VAL_6]] x !llvm.array<1024 x i32> {bindc_name = "d", in_type = !fir.array<1024xi32>, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_dataEd"} : (i64) -> !llvm.ptr<array<1024 x i32>>
// CHECK:           omp.target_enter_data   map((to -> %[[VAL_1]] : !llvm.ptr<array<1024 x i32>>), (to -> %[[VAL_3]] : !llvm.ptr<array<1024 x i32>>), (always, alloc -> %[[VAL_5]] : !llvm.ptr<array<1024 x i32>>))
// CHECK:           omp.target_exit_data   map((from -> %[[VAL_1]] : !llvm.ptr<array<1024 x i32>>), (from -> %[[VAL_3]] : !llvm.ptr<array<1024 x i32>>), (release -> %[[VAL_5]] : !llvm.ptr<array<1024 x i32>>), (always, delete -> %[[VAL_7]] : !llvm.ptr<array<1024 x i32>>))
// CHECK:           llvm.return
// CHECK:         }

// -----

func.func @_QPopenmp_target_data_region() {
  %0 = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFopenmp_target_data_regionEa"}
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFopenmp_target_data_regionEi"}
  omp.target_data   map((tofrom -> %0 : !fir.ref<!fir.array<1024xi32>>)) {
    %c1_i32 = arith.constant 1 : i32
    %2 = fir.convert %c1_i32 : (i32) -> index
    %c1024_i32 = arith.constant 1024 : i32
    %3 = fir.convert %c1024_i32 : (i32) -> index
    %c1 = arith.constant 1 : index
    %4 = fir.convert %2 : (index) -> i32
    %5:2 = fir.do_loop %arg0 = %2 to %3 step %c1 iter_args(%arg1 = %4) -> (index, i32) {
      fir.store %arg1 to %1 : !fir.ref<i32>
      %6 = fir.load %1 : !fir.ref<i32>
      %7 = fir.load %1 : !fir.ref<i32>
      %8 = fir.convert %7 : (i32) -> i64
      %c1_i64 = arith.constant 1 : i64
      %9 = arith.subi %8, %c1_i64 : i64
      %10 = fir.coordinate_of %0, %9 : (!fir.ref<!fir.array<1024xi32>>, i64) -> !fir.ref<i32>
      fir.store %6 to %10 : !fir.ref<i32>
      %11 = arith.addi %arg0, %c1 : index
      %12 = fir.convert %c1 : (index) -> i32
      %13 = fir.load %1 : !fir.ref<i32>
      %14 = arith.addi %13, %12 : i32
      fir.result %11, %14 : index, i32
    }
    fir.store %5#1 to %1 : !fir.ref<i32>
    omp.terminator
  }
  return
}

// CHECK-LABEL:   llvm.func @_QPopenmp_target_data_region() {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<1024 x i32> {bindc_name = "a", in_type = !fir.array<1024xi32>, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_data_regionEa"} : (i64) -> !llvm.ptr<array<1024 x i32>>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x i32 {bindc_name = "i", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFopenmp_target_data_regionEi"} : (i64) -> !llvm.ptr<i32>
// CHECK:           omp.target_data   map((tofrom -> %[[VAL_1]] : !llvm.ptr<array<1024 x i32>>)) {
// CHECK:             %[[VAL_4:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:             %[[VAL_5:.*]] = llvm.sext %[[VAL_4]] : i32 to i64
// CHECK:             %[[VAL_6:.*]] = llvm.mlir.constant(1024 : i32) : i32
// CHECK:             %[[VAL_7:.*]] = llvm.sext %[[VAL_6]] : i32 to i64
// CHECK:             %[[VAL_8:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_9:.*]] = llvm.trunc %[[VAL_5]] : i64 to i32
// CHECK:             %[[VAL_10:.*]] = llvm.sub %[[VAL_7]], %[[VAL_5]]  : i64
// CHECK:             %[[VAL_11:.*]] = llvm.add %[[VAL_10]], %[[VAL_8]]  : i64
// CHECK:             llvm.br ^bb1(%[[VAL_5]], %[[VAL_9]], %[[VAL_11]] : i64, i32, i64)
// CHECK:           ^bb1(%[[VAL_12:.*]]: i64, %[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i64):
// CHECK:             %[[VAL_15:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:             %[[VAL_16:.*]] = llvm.icmp "sgt" %[[VAL_14]], %[[VAL_15]] : i64
// CHECK:             llvm.cond_br %[[VAL_16]], ^bb2, ^bb3
// CHECK:           ^bb2:
// CHECK:             llvm.store %[[VAL_13]], %[[VAL_3]] : !llvm.ptr<i32>
// CHECK:             %[[VAL_17:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i32>
// CHECK:             %[[VAL_18:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i32>
// CHECK:             %[[VAL_19:.*]] = llvm.sext %[[VAL_18]] : i32 to i64
// CHECK:             %[[VAL_20:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_21:.*]] = llvm.sub %[[VAL_19]], %[[VAL_20]]  : i64
// CHECK:             %[[VAL_22:.*]] = llvm.getelementptr %[[VAL_1]][0, %[[VAL_21]]] : (!llvm.ptr<array<1024 x i32>>, i64) -> !llvm.ptr<i32>
// CHECK:             llvm.store %[[VAL_17]], %[[VAL_22]] : !llvm.ptr<i32>
// CHECK:             %[[VAL_23:.*]] = llvm.add %[[VAL_12]], %[[VAL_8]]  : i64
// CHECK:             %[[VAL_24:.*]] = llvm.trunc %[[VAL_8]] : i64 to i32
// CHECK:             %[[VAL_25:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<i32>
// CHECK:             %[[VAL_26:.*]] = llvm.add %[[VAL_25]], %[[VAL_24]]  : i32
// CHECK:             %[[VAL_27:.*]] = llvm.add %[[VAL_12]], %[[VAL_8]]  : i64
// CHECK:             %[[VAL_28:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_29:.*]] = llvm.sub %[[VAL_14]], %[[VAL_28]]  : i64
// CHECK:             llvm.br ^bb1(%[[VAL_27]], %[[VAL_26]], %[[VAL_29]] : i64, i32, i64)
// CHECK:           ^bb3:
// CHECK:             llvm.store %[[VAL_13]], %[[VAL_3]] : !llvm.ptr<i32>
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }

// -----

func.func @_QPomp_target_data_empty() {
  %0 = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFomp_target_data_emptyEa"}
  omp.target_data use_device_addr(%0 : !fir.ref<!fir.array<1024xi32>>) {
  }
  return
}

// CHECK-LABEL:   llvm.func @_QPomp_target_data_empty
// CHECK: omp.target_data   use_device_addr(%1 : !llvm.ptr<array<1024 x i32>>) {
// CHECK: }

// -----

func.func @_QPomp_target() {
  %0 = fir.alloca !fir.array<512xi32> {bindc_name = "a", uniq_name = "_QFomp_targetEa"}
  %c64_i32 = arith.constant 64 : i32
  omp.target   thread_limit(%c64_i32 : i32) map((tofrom -> %0 : !fir.ref<!fir.array<512xi32>>)) {
    %c10_i32 = arith.constant 10 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1_i64_0 = arith.constant 1 : i64
    %1 = arith.subi %c1_i64, %c1_i64_0 : i64
    %2 = fir.coordinate_of %0, %1 : (!fir.ref<!fir.array<512xi32>>, i64) -> !fir.ref<i32>
    fir.store %c10_i32 to %2 : !fir.ref<i32>
    omp.terminator
  }
  return
}

// CHECK-LABEL:   llvm.func @_QPomp_target() {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<512 x i32> {bindc_name = "a", in_type = !fir.array<512xi32>, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_targetEa"} : (i64) -> !llvm.ptr<array<512 x i32>>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(64 : i32) : i32
// CHECK:           omp.target   thread_limit(%[[VAL_2]] : i32) map((tofrom -> %[[VAL_1]] : !llvm.ptr<array<512 x i32>>)) {
// CHECK:             %[[VAL_3:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:             %[[VAL_4:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_5:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_6:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:             %[[VAL_7:.*]] = llvm.getelementptr %[[VAL_1]][0, %[[VAL_6]]] : (!llvm.ptr<array<512 x i32>>, i64) -> !llvm.ptr<i32>
// CHECK:             llvm.store %[[VAL_3]], %[[VAL_7]] : !llvm.ptr<i32>
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }

// -----

func.func @_QPsimdloop_with_nested_loop() {
  %0 = fir.alloca i32 {adapt.valuebyref}
  %1 = fir.alloca !fir.array<10xi32> {bindc_name = "a", uniq_name = "_QFsimdloop_with_nested_loopEa"}
  %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimdloop_with_nested_loopEi"}
  %3 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFsimdloop_with_nested_loopEj"}
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c1_i32_0 = arith.constant 1 : i32
  omp.simdloop   for  (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32_0) {
    fir.store %arg0 to %0 : !fir.ref<i32>
    %c1_i32_1 = arith.constant 1 : i32
    %4 = fir.convert %c1_i32_1 : (i32) -> index
    %c10_i32_2 = arith.constant 10 : i32
    %5 = fir.convert %c10_i32_2 : (i32) -> index
    %c1 = arith.constant 1 : index
    %6 = fir.do_loop %arg1 = %4 to %5 step %c1 -> index {
      %8 = fir.convert %arg1 : (index) -> i32
      fir.store %8 to %3 : !fir.ref<i32>
      %9 = fir.load %0 : !fir.ref<i32>
      %10 = fir.load %0 : !fir.ref<i32>
      %11 = fir.convert %10 : (i32) -> i64
      %c1_i64 = arith.constant 1 : i64
      %12 = arith.subi %11, %c1_i64 : i64
      %13 = fir.coordinate_of %1, %12 : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
      fir.store %9 to %13 : !fir.ref<i32>
      %14 = arith.addi %arg1, %c1 : index
      fir.result %14 : index
    }
    %7 = fir.convert %6 : (index) -> i32
    fir.store %7 to %3 : !fir.ref<i32>
    omp.yield
  }
  return
}

// CHECK-LABEL:   llvm.func @_QPsimdloop_with_nested_loop() {
// CHECK:           %[[LOWER:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[UPPER:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           %[[STEP:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           omp.simdloop   for  (%[[CNT:.*]]) : i32 = (%[[LOWER]]) to (%[[UPPER]]) inclusive step (%[[STEP]]) {
// CHECK:             llvm.br ^bb1(%[[VAL_1:.*]], %[[VAL_2:.*]] : i64, i64)
// CHECK:           ^bb1(%[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64):
// CHECK:             %[[VAL_5:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:             %[[VAL_6:.*]] = llvm.icmp "sgt" %[[VAL_4]], %[[VAL_5]] : i64
// CHECK:             llvm.cond_br %[[VAL_6]], ^bb2, ^bb3
// CHECK:           ^bb2:
// CHECK:             llvm.br ^bb1(%[[VAL_7:.*]], %[[VAL_8:.*]] : i64, i64)
// CHECK:           ^bb3:
// CHECK:             omp.yield
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }

// -----

func.func @_QPomp_taskgroup() {
  omp.taskgroup {
    omp.task   {
      fir.call @_QPwork() : () -> ()
      omp.terminator
    }
    omp.terminator
  }
  return
}
func.func private @_QPwork()

// CHECK-LABEL: llvm.func @_QPomp_taskgroup() {
// CHECK:          omp.taskgroup   {
// CHECK:            omp.task   {
// CHECK:              llvm.call @_QPwork() : () -> ()
// CHECK:              omp.terminator
// CHECK:            }
// CHECK:            omp.terminator
// CHECK:          }
// CHECK:          llvm.return
// CHECK:        }
// CHECK:        llvm.func @_QPwork() attributes {sym_visibility = "private"}
// CHECK:      }

// -----


func.func @_QQmain() {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %0 = fir.alloca i32
  omp.taskgroup   {
    %1 = fir.convert %c1 : (index) -> i32
    cf.br ^bb1(%1, %c5 : i32, index)
  ^bb1(%2: i32, %3: index):  // 2 preds: ^bb0, ^bb2
    %4 = arith.cmpi sgt, %3, %c0 : index
    cf.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    fir.store %2 to %0 : !fir.ref<i32>
    omp.task   {
      fir.call @_QFPdo_work(%0) : (!fir.ref<i32>) -> ()
      omp.terminator
    }
    %5 = fir.load %0 : !fir.ref<i32>
    %6 = arith.addi %5, %1 : i32
    %7 = arith.subi %3, %c1 : index
    cf.br ^bb1(%6, %7 : i32, index)
  ^bb3:  // pred: ^bb1
    fir.store %2 to %0 : !fir.ref<i32>
    omp.terminator
  }
  return
}
func.func private @_QFPdo_work(!fir.ref<i32>)

// CHECK-LABEL: llvm.func @_QQmain
// CHECK:          omp.taskgroup   {
// CHECK:            omp.task   {
// CHECK:              llvm.call @_QFPdo_work({{.*}}) : (!llvm.ptr<i32>) -> ()
// CHECK:              omp.terminator
// CHECK:            }
// CHECK:            omp.terminator
// CHECK:          }
// CHECK:          llvm.return
// CHECK:        }
// CHECK:        llvm.func @_QFPdo_work(!llvm.ptr<i32>)
// CHECK:      }

// -----

func.func @_QPs() {
  %0 = fir.address_of(@_QFsEc) : !fir.ref<i32>
  omp.atomic.update   %0 : !fir.ref<i32> {
  ^bb0(%arg0: i32):
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.addi %arg0, %c1_i32 : i32
    omp.yield(%1 : i32)
  }
  return
}
fir.global internal @_QFsEc : i32 {
  %c10_i32 = arith.constant 10 : i32
  fir.has_value %c10_i32 : i32
}

// CHECK-LABEL:  llvm.func @_QPs() {
// CHECK:    %[[GLOBAL_VAR:.*]] = llvm.mlir.addressof @[[GLOBAL:.*]] : !llvm.ptr<i32>
// CHECK:    omp.atomic.update   %[[GLOBAL_VAR]] : !llvm.ptr<i32> {
// CHECK:    ^bb0(%[[IN_VAL:.*]]: i32):
// CHECK:      %[[CONST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:      %[[OUT_VAL:.*]] = llvm.add %[[IN_VAL]], %[[CONST_1]]  : i32
// CHECK:      omp.yield(%[[OUT_VAL]] : i32)
// CHECK:    }
// CHECK:    llvm.return
// CHECK:  }
// CHECK:  llvm.mlir.global internal @[[GLOBAL]]() {{.*}} : i32 {
// CHECK:    %[[INIT_10:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:    llvm.return %[[INIT_10]] : i32
// CHECK:  }

func.func @_QPsb() {
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsbEi"}
  %1 = fir.alloca i32 {bindc_name = "li", uniq_name = "_QFsbEli"}
  fir.store %c0_i32 to %1 : !fir.ref<i32>
  omp.sections   {
    omp.section {
      %2 = fir.convert %c1 : (index) -> i32
      %3:2 = fir.do_loop %arg0 = %c1 to %c10 step %c1 iter_args(%arg1 = %2) -> (index, i32) {
        fir.store %arg1 to %0 : !fir.ref<i32>
        %4 = fir.load %1 : !fir.ref<i32>
        %5 = arith.addi %4, %c1_i32 : i32
        fir.store %5 to %1 : !fir.ref<i32>
        %6 = arith.addi %arg0, %c1 : index
        %7 = fir.convert %c1 : (index) -> i32
        %8 = fir.load %0 : !fir.ref<i32>
        %9 = arith.addi %8, %7 : i32
        fir.result %6, %9 : index, i32
      }
      fir.store %3#1 to %0 : !fir.ref<i32>
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK:  llvm.func @_QPsb() {
// CHECK:    %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:    %[[SIZE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[LI_REF:.*]] = llvm.alloca %6 x i32 {bindc_name = "li", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFsbEli"} : (i64) -> !llvm.ptr<i32>
// CHECK:    omp.sections   {
// CHECK:      omp.section {
// CHECK:        llvm.br ^[[BB_ENTRY:.*]]({{.*}})
// CHECK:      ^[[BB_ENTRY]]({{.*}}):
// CHECK:        %[[EXIT_COND:.*]] = llvm.icmp "sgt"
// CHECK:        llvm.cond_br %[[EXIT_COND]], ^[[BB_LOOP_BODY:.*]], ^[[BB_EXIT:.*]]
// CHECK:      ^[[BB_LOOP_BODY]]:
// CHECK:        %[[LI_VAL:.*]] = llvm.load %[[LI_REF]] : !llvm.ptr<i32>
// CHECK:        %[[LI_INC:.*]] = llvm.add %[[LI_VAL]], %[[ONE]]  : i32
// CHECK:        llvm.store %[[LI_INC]], %[[LI_REF]] : !llvm.ptr<i32>
// CHECK:        llvm.br ^[[BB_ENTRY]]({{.*}})
// CHECK:      ^[[BB_EXIT]]:
// CHECK:        omp.terminator
// CHECK:      }
// CHECK:      omp.terminator
// CHECK:    }
// CHECK:    llvm.return
// CHECK:  }

// -----

// CHECK:  omp.reduction.declare @[[EQV_REDUCTION:.*]] : i32 init {
// CHECK:  ^bb0(%{{.*}}: i32):
// CHECK:    %[[TRUE:.*]] = llvm.mlir.constant(1 : i64) : i32
// CHECK:    omp.yield(%[[TRUE]] : i32)
// CHECK:  } combiner {
// CHECK:  ^bb0(%[[ARG_1:.*]]: i32, %[[ARG_2:.*]]: i32):
// CHECK:    %[[ZERO_1:.*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK:    %[[ARGVAL_1:.*]] = llvm.icmp "ne" %[[ARG_1]], %[[ZERO_1]] : i32
// CHECK:    %[[ZERO_2:.*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK:    %[[ARGVAL_2:.*]] = llvm.icmp "ne" %[[ARG_2]], %[[ZERO_2]] : i32
// CHECK:    %[[RES:.*]] = llvm.icmp "eq" %[[ARGVAL_1]], %[[ARGVAL_2]] : i1
// CHECK:    %[[RES_EXT:.*]] = llvm.zext %[[RES]] : i1 to i32
// CHECK:    omp.yield(%[[RES_EXT]] : i32)
// CHECK:  }
// CHECK-LABEL:  @_QPsimple_reduction
// CHECK-SAME: %[[ARRAY_REF:.*]]: !llvm.ptr<array<100 x i32>>
// CHECK:    %[[RED_ACCUMULATOR:.*]] = llvm.alloca %2 x i32 {bindc_name = "x", in_type = !fir.logical<4>, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFsimple_reductionEx"} : (i64) -> !llvm.ptr<i32>
// CHECK:    omp.parallel   {
// CHECK:      omp.wsloop   reduction(@[[EQV_REDUCTION]] -> %[[RED_ACCUMULATOR]] : !llvm.ptr<i32>) for
// CHECK:        %[[ARRAY_ELEM_REF:.*]] = llvm.getelementptr %[[ARRAY_REF]][0, %{{.*}}] : (!llvm.ptr<array<100 x i32>>, i64) -> !llvm.ptr<i32>
// CHECK:        %[[ARRAY_ELEM:.*]] = llvm.load %[[ARRAY_ELEM_REF]] : !llvm.ptr<i32>
// CHECK:        omp.reduction %[[ARRAY_ELEM]], %[[RED_ACCUMULATOR]] : i32, !llvm.ptr<i32>
// CHECK:        omp.yield
// CHECK:      omp.terminator
// CHECK:    llvm.return

omp.reduction.declare @eqv_reduction : !fir.logical<4> init {
^bb0(%arg0: !fir.logical<4>):
  %true = arith.constant true
  %0 = fir.convert %true : (i1) -> !fir.logical<4>
  omp.yield(%0 : !fir.logical<4>)
} combiner {
^bb0(%arg0: !fir.logical<4>, %arg1: !fir.logical<4>):
  %0 = fir.convert %arg0 : (!fir.logical<4>) -> i1
  %1 = fir.convert %arg1 : (!fir.logical<4>) -> i1
  %2 = arith.cmpi eq, %0, %1 : i1
  %3 = fir.convert %2 : (i1) -> !fir.logical<4>
  omp.yield(%3 : !fir.logical<4>)
}
func.func @_QPsimple_reduction(%arg0: !fir.ref<!fir.array<100x!fir.logical<4>>> {fir.bindc_name = "y"}) {
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimple_reductionEi"}
  %1 = fir.alloca !fir.logical<4> {bindc_name = "x", uniq_name = "_QFsimple_reductionEx"}
  %true = arith.constant true
  %2 = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %2 to %1 : !fir.ref<!fir.logical<4>>
  omp.parallel   {
    %3 = fir.alloca i32 {adapt.valuebyref, pinned}
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32_0 = arith.constant 1 : i32
    omp.wsloop   reduction(@eqv_reduction -> %1 : !fir.ref<!fir.logical<4>>) for  (%arg1) : i32 = (%c1_i32) to (%c100_i32) inclusive step (%c1_i32_0) {
      fir.store %arg1 to %3 : !fir.ref<i32>
      %4 = fir.load %3 : !fir.ref<i32>
      %5 = fir.convert %4 : (i32) -> i64
      %c1_i64 = arith.constant 1 : i64
      %6 = arith.subi %5, %c1_i64 : i64
      %7 = fir.coordinate_of %arg0, %6 : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
      %8 = fir.load %7 : !fir.ref<!fir.logical<4>>
      omp.reduction %8, %1 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
      omp.yield
    }
    omp.terminator
  }
  return
}

// -----

// CHECK: llvm.func @_QPs
// CHECK: omp.atomic.read %{{.*}} = %{{.*}}   : !llvm.ptr<struct<(f32, f32)>>, !llvm.struct<(f32, f32)>

func.func @_QPs(%arg0: !fir.ref<!fir.complex<4>> {fir.bindc_name = "x"}) {
  %0 = fir.alloca !fir.complex<4> {bindc_name = "v", uniq_name = "_QFsEv"}
  omp.atomic.read %0 = %arg0   : !fir.ref<!fir.complex<4>>, !fir.complex<4>
  return
}

// -----

// Test if llvm.alloca is properly inserted in the omp section

//CHECK:  %[[CONST:.*]] = llvm.mlir.constant(1 : i64) : i64
//CHECK:  %[[ALLOCA:.*]] = llvm.alloca %[[CONST]] x !llvm.struct<(ptr<i32>, i64, i32, i8, i8, i8, i8)> {bindc_name = "iattr", in_type = !fir.box<!fir.ptr<i32>>, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFEiattr"} : (i64) -> !llvm.ptr<struct<(ptr<i32>, i64, i32, i8, i8, i8, i8)>>
//CHECK:  omp.parallel   {
//CHECK:    %[[CONST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
//CHECK:    %[[ALLOCA_1:.*]] = llvm.alloca %[[CONST_1:.*]] x !llvm.struct<(ptr<i32>, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<struct<(ptr<i32>, i64, i32, i8, i8, i8, i8)>>
//CHECK:    %[[LOAD:.*]] = llvm.load %[[ALLOCA]] : !llvm.ptr<struct<(ptr<i32>, i64, i32, i8, i8, i8, i8)>>
//CHECK:    llvm.store %[[LOAD]], %[[ALLOCA_1]] : !llvm.ptr<struct<(ptr<i32>, i64, i32, i8, i8, i8, i8)>>
//CHECK:    %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA_1]][0, 0] : (!llvm.ptr<struct<(ptr<i32>, i64, i32, i8, i8, i8, i8)>>) -> !llvm.ptr<ptr<i32>>
//CHECK:    %[[LOAD_2:.*]] = llvm.load %[[GEP]] : !llvm.ptr<ptr<i32>>
//CHECK:    omp.terminator
//CHECK:  }

func.func @_QQmain() attributes {fir.bindc_name = "mn"} {
  %0 = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "iattr", uniq_name = "_QFEiattr"}
  %1 = fir.zero_bits !fir.ptr<i32>
  %2 = fir.embox %1 : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
  fir.store %2 to %0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %3 = fir.address_of(@_QFEx) : !fir.ref<i32>
  %4 = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
  %5 = fir.embox %3 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
  fir.store %5 to %0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  omp.parallel   {
    %6 = fir.load %4 : !fir.ref<i32>
    %7 = fir.load %0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
    %8 = fir.box_addr %7 : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
    fir.store %6 to %8 : !fir.ptr<i32>
    omp.terminator
  }
  return
}
fir.global internal @_QFEx target : i32 {
  %0 = fir.zero_bits i32
  fir.has_value %0 : i32
}

// -----

// Test if llvm.alloca is properly inserted in the omp ordered region

// CHECK: llvm.func @sub_
func.func @sub_() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsubEi"}
// CHECK: omp.ordered_region
  omp.ordered_region {
    %1 = fir.convert %c1 : (index) -> i32
    cf.br ^bb1(%1, %c1 : i32, index)
  ^bb1(%2: i32, %3: index):  // 2 preds: ^bb0, ^bb2
    %4 = arith.cmpi sgt, %3, %c0 : index
    cf.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    fir.store %2 to %0 : !fir.ref<i32>
    %5 = fir.load %0 : !fir.ref<i32>
// CHECK: llvm.add
    %6 = arith.addi %5, %1 : i32
// CHECK: llvm.sub
    %7 = arith.subi %3, %c1 : index
    cf.br ^bb1(%6, %7 : i32, index)
  ^bb3:  // pred: ^bb1
    fir.store %2 to %0 : !fir.ref<i32>
// CHECK: omp.terminator
    omp.terminator
  }
  return
} 
