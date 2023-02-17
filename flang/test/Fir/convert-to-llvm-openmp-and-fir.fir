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
// CHECK:      %[[I_VAR:.*]] = llvm.alloca %[[ONE_3]] x i32 {adapt.valuebyref, in_type = i32, operand_segment_sizes = array<i32: 0, 0>, pinned} : (i64) -> !llvm.ptr<i32>
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
// CHECK:      %[[I_VAR:.*]] = llvm.alloca %[[ONE_3]] x i32 {adapt.valuebyref, in_type = i32, operand_segment_sizes = array<i32: 0, 0>, pinned} : (i64) -> !llvm.ptr<i32>
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
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<1024 x i32> {bindc_name = "a", in_type = !fir.array<1024xi32>, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_dataEa"} : (i64) -> !llvm.ptr<array<1024 x i32>>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.array<1024 x i32> {bindc_name = "b", in_type = !fir.array<1024xi32>, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_dataEb"} : (i64) -> !llvm.ptr<array<1024 x i32>>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_4]] x !llvm.array<1024 x i32> {bindc_name = "c", in_type = !fir.array<1024xi32>, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_dataEc"} : (i64) -> !llvm.ptr<array<1024 x i32>>
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_7:.*]] = llvm.alloca %[[VAL_6]] x !llvm.array<1024 x i32> {bindc_name = "d", in_type = !fir.array<1024xi32>, operand_segment_sizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_dataEd"} : (i64) -> !llvm.ptr<array<1024 x i32>>
// CHECK:           omp.target_enter_data   map((to -> %[[VAL_1]] : !llvm.ptr<array<1024 x i32>>), (to -> %[[VAL_3]] : !llvm.ptr<array<1024 x i32>>), (always, alloc -> %[[VAL_5]] : !llvm.ptr<array<1024 x i32>>))
// CHECK:           omp.target_exit_data   map((from -> %[[VAL_1]] : !llvm.ptr<array<1024 x i32>>), (from -> %[[VAL_3]] : !llvm.ptr<array<1024 x i32>>), (release -> %[[VAL_5]] : !llvm.ptr<array<1024 x i32>>), (always, delete -> %[[VAL_7]] : !llvm.ptr<array<1024 x i32>>))
// CHECK:           llvm.return
// CHECK:         }

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
