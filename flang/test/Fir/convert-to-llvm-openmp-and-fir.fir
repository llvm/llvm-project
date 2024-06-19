// RUN: fir-opt --split-input-file --cfg-conversion --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s

func.func @_QPsb1(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr"}) {
  %c1_i64 = arith.constant 1 : i64
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsbEi"}
  omp.parallel  {
    %1 = fir.alloca i32 {adapt.valuebyref, pinned}
    %2 = fir.load %arg0 : !fir.ref<i32>
    omp.wsloop nowait {
      omp.loop_nest (%arg2) : i32 = (%c1_i32) to (%2) inclusive step (%c1_i32)  {
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
    omp.terminator
  }
  return
}

// CHECK-LABEL:  _QPsb1
// CHECK-SAME: %[[N_REF:.*]]: !llvm.ptr {fir.bindc_name = "n"}, %[[ARR_REF:.*]]: !llvm.ptr {fir.bindc_name = "arr"}) {
// CHECK:    %[[ONE_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[ONE_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: omp.parallel   {
// CHECK:      %[[ONE_3:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:      %[[I_VAR:.*]] = llvm.alloca %[[ONE_3]] x i32 {pinned} : (i64) -> !llvm.ptr
// CHECK:      %[[N:.*]] = llvm.load %[[N_REF]] : !llvm.ptr -> i32
// CHECK: omp.wsloop nowait
// CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[ONE_2]]) to (%[[N]]) inclusive step (%[[ONE_2]]) {
// CHECK:   llvm.store %[[I]], %[[I_VAR]] : i32, !llvm.ptr
// CHECK:   %[[I1:.*]] = llvm.load %[[I_VAR]] : !llvm.ptr -> i32
// CHECK:   %[[I1_EXT:.*]] = llvm.sext %[[I1]] : i32 to i64
// CHECK:   %[[I_CSTYLE:.*]] = llvm.sub %[[I1_EXT]], %[[ONE_1]]  : i64
// CHECK:   %[[ARR_I_REF:.*]] = llvm.getelementptr %[[ARR_REF]][%[[I_CSTYLE]]] : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK:   llvm.store %[[I1]], %[[ARR_I_REF]] : i32, !llvm.ptr
// CHECK: omp.yield
// CHECK: }
// CHECK: omp.terminator
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
// CHECK-SAME: %[[X_REF:.*]]: !llvm.ptr {fir.bindc_name = "x"}, %[[N_REF:.*]]: !llvm.ptr {fir.bindc_name = "n"}) {
// CHECK: omp.parallel   {
// CHECK:   omp.master {
// CHECK:     %[[N:.*]] = llvm.load %[[N_REF]] : !llvm.ptr -> i32
// CHECK:     llvm.store %[[N]], %[[X_REF]] : i32, !llvm.ptr
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
    omp.wsloop {
      omp.loop_nest (%indx) : i32 = (%c1) to (%c50) inclusive step (%c1) {
        %1 = fir.convert %indx : (i32) -> i64
        %c1_i64 = arith.constant 1 : i64
        %2 = arith.subi %1, %c1_i64 : i64
        %3 = fir.coordinate_of %arr, %2 : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
        fir.store %indx to %3 : !fir.ref<i32>
        omp.yield
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// Check only for the structure of the OpenMP portion and the feasibility of the conversion
// CHECK-LABEL: @_QPsb
// CHECK-SAME: %{{.*}}: !llvm.ptr {fir.bindc_name = "arr"}
// CHECK:    omp.parallel   {
// CHECK:      %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:      %[[C50:.*]] = llvm.mlir.constant(50 : i32) : i32
// CHECK:      omp.wsloop {
// CHECK-NEXT:   omp.loop_nest (%[[INDX:.*]]) : i32 = (%[[C1]]) to (%[[C50]]) inclusive step (%[[C1]]) {
// CHECK:          llvm.store %[[INDX]], %{{.*}} : i32, !llvm.ptr
// CHECK:          omp.yield
// CHECK:        omp.terminator
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
// CHECK-SAME:            (%[[ARG0:.+]]: !llvm.ptr {fir.bindc_name = "a"}, %[[ARG1:.+]]: !llvm.ptr {fir.bindc_name = "b"})
// CHECK: omp.sections {
// CHECK:   omp.section {
// CHECK:     llvm.call @foo(%arg0) : (!llvm.ptr) -> ()
// CHECK:     omp.terminator
// CHECK:   }
// CHECK:   omp.section {
// CHECK:     llvm.call @bar(%[[ARG0]], %[[ARG1]]) : (!llvm.ptr, !llvm.ptr) -> ()
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
    omp.simd {
      omp.loop_nest (%arg2) : i32 = (%c1_i32) to (%2) step (%c1_i32) {
        fir.store %arg2 to %1 : !fir.ref<i32>
        %3 = fir.load %1 : !fir.ref<i32>
        %4 = fir.convert %3 : (i32) -> i64
        %5 = arith.subi %4, %c1_i64 : i64
        %6 = fir.coordinate_of %arg1, %5 : (!fir.ref<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
        fir.store %3 to %6 : !fir.ref<i32>
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL:  _QPsimd1
// CHECK-SAME: %[[N_REF:.*]]: !llvm.ptr {fir.bindc_name = "n"}, %[[ARR_REF:.*]]: !llvm.ptr {fir.bindc_name = "arr"}) {
// CHECK:    %[[ONE_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[ONE_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: omp.parallel   {
// CHECK:      %[[ONE_3:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:      %[[I_VAR:.*]] = llvm.alloca %[[ONE_3]] x i32 {pinned} : (i64) -> !llvm.ptr
// CHECK:      %[[N:.*]] = llvm.load %[[N_REF]] : !llvm.ptr -> i32
// CHECK: omp.simd {
// CHECK-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[ONE_2]]) to (%[[N]]) step (%[[ONE_2]]) {
// CHECK:   llvm.store %[[I]], %[[I_VAR]] : i32, !llvm.ptr
// CHECK:   %[[I1:.*]] = llvm.load %[[I_VAR]] : !llvm.ptr -> i32
// CHECK:   %[[I1_EXT:.*]] = llvm.sext %[[I1]] : i32 to i64
// CHECK:   %[[I_CSTYLE:.*]] = llvm.sub %[[I1_EXT]], %[[ONE_1]]  : i64
// CHECK:   %[[ARR_I_REF:.*]] = llvm.getelementptr %[[ARR_REF]][%[[I_CSTYLE]]] : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK:   llvm.store %[[I1]], %[[ARR_I_REF]] : i32, !llvm.ptr
// CHECK: omp.yield
// CHECK: }
// CHECK: }
// CHECK: omp.terminator
// CHECK: }
// CHECK: llvm.return
// CHECK: }

// -----

func.func @_QPomp_target_data() {
  %c1024 = arith.constant 1024 : index
  %0 = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFomp_target_dataEa"}
  %c1024_0 = arith.constant 1024 : index
  %1 = fir.alloca !fir.array<1024xi32> {bindc_name = "b", uniq_name = "_QFomp_target_dataEb"}
  %c1024_1 = arith.constant 1024 : index
  %2 = fir.alloca !fir.array<1024xi32> {bindc_name = "c", uniq_name = "_QFomp_target_dataEc"}
  %c1024_2 = arith.constant 1024 : index
  %3 = fir.alloca !fir.array<1024xi32> {bindc_name = "d", uniq_name = "_QFomp_target_dataEd"}
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %4 = arith.subi %c1024, %c1 : index
  %5 = omp.map.bounds   lower_bound(%c0 : index) upper_bound(%4 : index) extent(%c1024 : index) stride(%c1 : index) start_idx(%c1 : index)
  %6 = omp.map.info var_ptr(%0 : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(to) capture(ByRef) bounds(%5) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
  %c1_3 = arith.constant 1 : index
  %c0_4 = arith.constant 0 : index
  %7 = arith.subi %c1024_0, %c1_3 : index
  %8 = omp.map.bounds   lower_bound(%c0_4 : index) upper_bound(%7 : index) extent(%c1024_0 : index) stride(%c1_3 : index) start_idx(%c1_3 : index)
  %9 = omp.map.info var_ptr(%1 : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(to) capture(ByRef) bounds(%8) -> !fir.ref<!fir.array<1024xi32>> {name = "b"}
  %c1_5 = arith.constant 1 : index
  %c0_6 = arith.constant 0 : index
  %10 = arith.subi %c1024_1, %c1_5 : index
  %11 = omp.map.bounds   lower_bound(%c0_6 : index) upper_bound(%10 : index) extent(%c1024_1 : index) stride(%c1_5 : index) start_idx(%c1_5 : index)
  %12 = omp.map.info var_ptr(%2 : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(always, exit_release_or_enter_alloc) capture(ByRef) bounds(%11) -> !fir.ref<!fir.array<1024xi32>> {name = "c"}
  omp.target_enter_data   map_entries(%6, %9, %12 : !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>)
  %c1_7 = arith.constant 1 : index
  %c0_8 = arith.constant 0 : index
  %13 = arith.subi %c1024, %c1_7 : index
  %14 = omp.map.bounds   lower_bound(%c0_8 : index) upper_bound(%13 : index) extent(%c1024 : index) stride(%c1_7 : index) start_idx(%c1_7 : index)
  %15 = omp.map.info var_ptr(%0 : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(from) capture(ByRef) bounds(%14) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
  %c1_9 = arith.constant 1 : index
  %c0_10 = arith.constant 0 : index
  %16 = arith.subi %c1024_0, %c1_9 : index
  %17 = omp.map.bounds   lower_bound(%c0_10 : index) upper_bound(%16 : index) extent(%c1024_0 : index) stride(%c1_9 : index) start_idx(%c1_9 : index)
  %18 = omp.map.info var_ptr(%1 : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(from) capture(ByRef) bounds(%17) -> !fir.ref<!fir.array<1024xi32>> {name = "b"}
  %c1_11 = arith.constant 1 : index
  %c0_12 = arith.constant 0 : index
  %19 = arith.subi %c1024_1, %c1_11 : index
  %20 = omp.map.bounds   lower_bound(%c0_12 : index) upper_bound(%19 : index) extent(%c1024_1 : index) stride(%c1_11 : index) start_idx(%c1_11 : index)
  %21 = omp.map.info var_ptr(%2 : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) bounds(%20) -> !fir.ref<!fir.array<1024xi32>> {name = "c"}
  %c1_13 = arith.constant 1 : index
  %c0_14 = arith.constant 0 : index
  %22 = arith.subi %c1024_2, %c1_13 : index
  %23 = omp.map.bounds   lower_bound(%c0_14 : index) upper_bound(%22 : index) extent(%c1024_2 : index) stride(%c1_13 : index) start_idx(%c1_13 : index)
  %24 = omp.map.info var_ptr(%3 : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(always, delete) capture(ByRef) bounds(%23) -> !fir.ref<!fir.array<1024xi32>> {name = "d"}
  omp.target_exit_data   map_entries(%15, %18, %21, %24 : !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>, !fir.ref<!fir.array<1024xi32>>)
  return
}

// CHECK-LABEL:   llvm.func @_QPomp_target_data() {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1024 : index) : i64
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.array<1024 x i32> {bindc_name = "d"} : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.array<1024 x i32> {bindc_name = "c"} : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_6:.*]] = llvm.alloca %[[VAL_5]] x !llvm.array<1024 x i32> {bindc_name = "b"} : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_8:.*]] = llvm.alloca %[[VAL_7]] x !llvm.array<1024 x i32> {bindc_name = "a"} : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(1024 : index) : i64
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(1024 : index) : i64
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(1024 : index) : i64
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(1023 : index) : i64
// CHECK:           %[[VAL_15:.*]] = omp.map.bounds lower_bound(%[[VAL_13]] : i64) upper_bound(%[[VAL_14]] : i64) extent(%[[VAL_0]] : i64) stride(%[[VAL_12]] : i64) start_idx(%[[VAL_12]] : i64)
// CHECK:           %[[VAL_16:.*]] = omp.map.info var_ptr(%[[VAL_8]] : !llvm.ptr, !llvm.array<1024 x i32>) map_clauses(to) capture(ByRef) bounds(%[[VAL_15]]) -> !llvm.ptr {name = "a"}
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(1023 : index) : i64
// CHECK:           %[[VAL_20:.*]] = omp.map.bounds lower_bound(%[[VAL_18]] : i64) upper_bound(%[[VAL_19]] : i64) extent(%[[VAL_9]] : i64) stride(%[[VAL_17]] : i64) start_idx(%[[VAL_17]] : i64)
// CHECK:           %[[VAL_21:.*]] = omp.map.info var_ptr(%[[VAL_6]] : !llvm.ptr, !llvm.array<1024 x i32>) map_clauses(to) capture(ByRef) bounds(%[[VAL_20]]) -> !llvm.ptr {name = "b"}
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(1023 : index) : i64
// CHECK:           %[[VAL_25:.*]] = omp.map.bounds lower_bound(%[[VAL_23]] : i64) upper_bound(%[[VAL_24]] : i64) extent(%[[VAL_10]] : i64) stride(%[[VAL_22]] : i64) start_idx(%[[VAL_22]] : i64)
// CHECK:           %[[VAL_26:.*]] = omp.map.info var_ptr(%[[VAL_4]] : !llvm.ptr, !llvm.array<1024 x i32>) map_clauses(always, exit_release_or_enter_alloc) capture(ByRef) bounds(%[[VAL_25]]) -> !llvm.ptr {name = "c"}
// CHECK:           omp.target_enter_data map_entries(%[[VAL_16]], %[[VAL_21]], %[[VAL_26]] : !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CHECK:           %[[VAL_27:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.constant(1023 : index) : i64
// CHECK:           %[[VAL_30:.*]] = omp.map.bounds lower_bound(%[[VAL_28]] : i64) upper_bound(%[[VAL_29]] : i64) extent(%[[VAL_0]] : i64) stride(%[[VAL_27]] : i64) start_idx(%[[VAL_27]] : i64)
// CHECK:           %[[VAL_31:.*]] = omp.map.info var_ptr(%[[VAL_8]] : !llvm.ptr, !llvm.array<1024 x i32>) map_clauses(from) capture(ByRef) bounds(%[[VAL_30]]) -> !llvm.ptr {name = "a"}
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_33:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(1023 : index) : i64
// CHECK:           %[[VAL_35:.*]] = omp.map.bounds lower_bound(%[[VAL_33]] : i64) upper_bound(%[[VAL_34]] : i64) extent(%[[VAL_9]] : i64) stride(%[[VAL_32]] : i64) start_idx(%[[VAL_32]] : i64)
// CHECK:           %[[VAL_36:.*]] = omp.map.info var_ptr(%[[VAL_6]] : !llvm.ptr, !llvm.array<1024 x i32>) map_clauses(from) capture(ByRef) bounds(%[[VAL_35]]) -> !llvm.ptr {name = "b"}
// CHECK:           %[[VAL_37:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_38:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_39:.*]] = llvm.mlir.constant(1023 : index) : i64
// CHECK:           %[[VAL_40:.*]] = omp.map.bounds lower_bound(%[[VAL_38]] : i64) upper_bound(%[[VAL_39]] : i64) extent(%[[VAL_10]] : i64) stride(%[[VAL_37]] : i64) start_idx(%[[VAL_37]] : i64)
// CHECK:           %[[VAL_41:.*]] = omp.map.info var_ptr(%[[VAL_4]] : !llvm.ptr, !llvm.array<1024 x i32>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) bounds(%[[VAL_40]]) -> !llvm.ptr {name = "c"}
// CHECK:           %[[VAL_42:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_44:.*]] = llvm.mlir.constant(1023 : index) : i64
// CHECK:           %[[VAL_45:.*]] = omp.map.bounds lower_bound(%[[VAL_43]] : i64) upper_bound(%[[VAL_44]] : i64) extent(%[[VAL_11]] : i64) stride(%[[VAL_42]] : i64) start_idx(%[[VAL_42]] : i64)
// CHECK:           %[[VAL_46:.*]] = omp.map.info var_ptr(%[[VAL_2]] : !llvm.ptr, !llvm.array<1024 x i32>) map_clauses(always, delete) capture(ByRef) bounds(%[[VAL_45]]) -> !llvm.ptr {name = "d"}
// CHECK:           omp.target_exit_data map_entries(%[[VAL_31]], %[[VAL_36]], %[[VAL_41]], %[[VAL_46]] : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CHECK:           llvm.return
// CHECK:         }

// -----

func.func @_QPopenmp_target_data_region() {
  %0 = fir.alloca !fir.array<1024xi32> {bindc_name = "a", uniq_name = "_QFopenmp_target_data_regionEa"}
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFopenmp_target_data_regionEi"}
  %c1024 = arith.constant 1024 : index
  %c3 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.subi %c1024, %c3 : index
  %bound = omp.map.bounds   lower_bound(%c0 : index) upper_bound(%c2 : index) extent(%c1024 : index) stride(%c3 : index) start_idx(%c3 : index)
  %entry = omp.map.info var_ptr(%0 : !fir.ref<!fir.array<1024xi32>>, !fir.array<1024xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%bound) -> !fir.ref<!fir.array<1024xi32>> {name = "a"}
  omp.target_data   map_entries(%entry : !fir.ref<!fir.array<1024xi32>>) {
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
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<1024 x i32> {bindc_name = "a"} : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_MAX:.*]] = llvm.mlir.constant(1024 : index) : i64
// CHECK:           %[[VAL_ONE:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_ZERO:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_UPPER:.*]] = llvm.mlir.constant(1023 : index) : i64
// CHECK:           %[[VAL_BOUNDS:.*]] = omp.map.bounds   lower_bound(%[[VAL_ZERO]] : i64) upper_bound(%[[VAL_UPPER]] : i64) extent(%[[VAL_MAX]] : i64) stride(%[[VAL_ONE]] : i64) start_idx(%[[VAL_ONE]] : i64)
// CHECK:           %[[VAL_MAP:.*]] = omp.map.info var_ptr(%[[VAL_1]] : !llvm.ptr, !llvm.array<1024 x i32>)  map_clauses(tofrom) capture(ByRef) bounds(%[[VAL_BOUNDS]]) -> !llvm.ptr {name = "a"}
// CHECK:           omp.target_data   map_entries(%[[VAL_MAP]] : !llvm.ptr) {
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
// CHECK:             llvm.store %[[VAL_13]], %[[VAL_3]] : i32, !llvm.ptr
// CHECK:             %[[VAL_17:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i32
// CHECK:             %[[VAL_18:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i32
// CHECK:             %[[VAL_19:.*]] = llvm.sext %[[VAL_18]] : i32 to i64
// CHECK:             %[[VAL_20:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_21:.*]] = llvm.sub %[[VAL_19]], %[[VAL_20]]  : i64
// CHECK:             %[[VAL_22:.*]] = llvm.getelementptr %[[VAL_1]][0, %[[VAL_21]]] : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK:             llvm.store %[[VAL_17]], %[[VAL_22]] : i32, !llvm.ptr
// CHECK:             %[[VAL_23:.*]] = llvm.add %[[VAL_12]], %[[VAL_8]]  : i64
// CHECK:             %[[VAL_24:.*]] = llvm.trunc %[[VAL_8]] : i64 to i32
// CHECK:             %[[VAL_25:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i32
// CHECK:             %[[VAL_26:.*]] = llvm.add %[[VAL_25]], %[[VAL_24]]  : i32
// CHECK:             %[[VAL_27:.*]] = llvm.add %[[VAL_12]], %[[VAL_8]]  : i64
// CHECK:             %[[VAL_28:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:             %[[VAL_29:.*]] = llvm.sub %[[VAL_14]], %[[VAL_28]]  : i64
// CHECK:             llvm.br ^bb1(%[[VAL_27]], %[[VAL_26]], %[[VAL_29]] : i64, i32, i64)
// CHECK:           ^bb3:
// CHECK:             llvm.store %[[VAL_13]], %[[VAL_3]] : i32, !llvm.ptr
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
// CHECK: omp.target_data   use_device_addr(%1 : !llvm.ptr) {
// CHECK: }

// -----

func.func @_QPomp_target() {
  %c512 = arith.constant 512 : index
  %0 = fir.alloca !fir.array<512xi32> {bindc_name = "a", uniq_name = "_QFomp_targetEa"}
  %c64_i32 = arith.constant 64 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %1 = arith.subi %c512, %c1 : index
  %2 = omp.map.bounds   lower_bound(%c0 : index) upper_bound(%1 : index) extent(%c512 : index) stride(%c1 : index) start_idx(%c1 : index)
  %3 = omp.map.info var_ptr(%0 : !fir.ref<!fir.array<512xi32>>, !fir.array<512xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%2) -> !fir.ref<!fir.array<512xi32>> {name = "a"}
  omp.target   thread_limit(%c64_i32 : i32) map_entries(%3 -> %arg0 : !fir.ref<!fir.array<512xi32>>) {
    ^bb0(%arg0: !fir.ref<!fir.array<512xi32>>):
    %c10_i32 = arith.constant 10 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1_i64_0 = arith.constant 1 : i64
    %4 = arith.subi %c1_i64, %c1_i64_0 : i64
    %5 = fir.coordinate_of %arg0, %4 : (!fir.ref<!fir.array<512xi32>>, i64) -> !fir.ref<i32>
    fir.store %c10_i32 to %5 : !fir.ref<i32>
    omp.terminator
  }
  return
}

// CHECK-LABEL:   llvm.func @_QPomp_target() {
// CHECK:           %[[EXTENT:.*]] = llvm.mlir.constant(512 : index) : i64
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<512 x i32> {bindc_name = "a"} : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(64 : i32) : i32
// CHECK:           %[[STRIDE:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[LOWER:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[UPPER:.*]] = llvm.mlir.constant(511 : index) : i64
// CHECK:           %[[BOUNDS:.*]] = omp.map.bounds   lower_bound(%[[LOWER]] : i64) upper_bound(%[[UPPER]] : i64) extent(%[[EXTENT]] : i64) stride(%[[STRIDE]] : i64) start_idx(%[[STRIDE]] : i64)
// CHECK:           %[[MAP:.*]] = omp.map.info var_ptr(%2 : !llvm.ptr, !llvm.array<512 x i32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !llvm.ptr {name = "a"}
// CHECK:           omp.target   thread_limit(%[[VAL_2]] : i32) map_entries(%[[MAP]] -> %[[ARG_0:.*]] : !llvm.ptr) {
// CHECK:           ^bb0(%[[ARG_0]]: !llvm.ptr):
// CHECK:             %[[VAL_3:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:             %[[VAL_4:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_5:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:             %[[VAL_6:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:             %[[VAL_7:.*]] = llvm.getelementptr %[[ARG_0]][0, %[[VAL_6]]] : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK:             llvm.store %[[VAL_3]], %[[VAL_7]] : i32, !llvm.ptr
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }

// -----

func.func @_QPsimd_with_nested_loop() {
  %0 = fir.alloca i32 {adapt.valuebyref}
  %1 = fir.alloca !fir.array<10xi32> {bindc_name = "a", uniq_name = "_QFsimd_with_nested_loopEa"}
  %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsimd_with_nested_loopEi"}
  %3 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFsimd_with_nested_loopEj"}
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c1_i32_0 = arith.constant 1 : i32
  omp.simd {
    omp.loop_nest (%arg0) : i32 = (%c1_i32) to (%c10_i32) inclusive step (%c1_i32_0) {
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
  }
  return
}

// CHECK-LABEL:   llvm.func @_QPsimd_with_nested_loop() {
// CHECK:           %[[LOWER:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[UPPER:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           %[[STEP:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           omp.simd {
// CHECK-NEXT:        omp.loop_nest (%[[CNT:.*]]) : i32 = (%[[LOWER]]) to (%[[UPPER]]) inclusive step (%[[STEP]]) {
// CHECK:               llvm.br ^bb1(%[[VAL_1:.*]], %[[VAL_2:.*]] : i64, i64)
// CHECK:             ^bb1(%[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64):
// CHECK:               %[[VAL_5:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:               %[[VAL_6:.*]] = llvm.icmp "sgt" %[[VAL_4]], %[[VAL_5]] : i64
// CHECK:               llvm.cond_br %[[VAL_6]], ^bb2, ^bb3
// CHECK:             ^bb2:
// CHECK:               llvm.br ^bb1(%[[VAL_7:.*]], %[[VAL_8:.*]] : i64, i64)
// CHECK:             ^bb3:
// CHECK:               omp.yield
// CHECK:             }
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
// CHECK:              llvm.call @_QFPdo_work({{.*}}) : (!llvm.ptr) -> ()
// CHECK:              omp.terminator
// CHECK:            }
// CHECK:            omp.terminator
// CHECK:          }
// CHECK:          llvm.return
// CHECK:        }
// CHECK:        llvm.func @_QFPdo_work(!llvm.ptr)
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
// CHECK:    %[[GLOBAL_VAR:.*]] = llvm.mlir.addressof @[[GLOBAL:.*]] : !llvm.ptr
// CHECK:    omp.atomic.update   %[[GLOBAL_VAR]] : !llvm.ptr {
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
// CHECK:    %[[SIZE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:    %[[LI_REF:.*]] = llvm.alloca %[[SIZE]] x i32 {bindc_name = "li"} : (i64) -> !llvm.ptr
// CHECK:    %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:    omp.sections   {
// CHECK:      omp.section {
// CHECK:        llvm.br ^[[BB_ENTRY:.*]]({{.*}})
// CHECK:      ^[[BB_ENTRY]]({{.*}}):
// CHECK:        %[[EXIT_COND:.*]] = llvm.icmp "sgt"
// CHECK:        llvm.cond_br %[[EXIT_COND]], ^[[BB_LOOP_BODY:.*]], ^[[BB_EXIT:.*]]
// CHECK:      ^[[BB_LOOP_BODY]]:
// CHECK:        %[[LI_VAL:.*]] = llvm.load %[[LI_REF]] : !llvm.ptr -> i32
// CHECK:        %[[LI_INC:.*]] = llvm.add %[[LI_VAL]], %[[ONE]]  : i32
// CHECK:        llvm.store %[[LI_INC]], %[[LI_REF]] : i32, !llvm.ptr
// CHECK:        llvm.br ^[[BB_ENTRY]]({{.*}})
// CHECK:      ^[[BB_EXIT]]:
// CHECK:        omp.terminator
// CHECK:      }
// CHECK:      omp.terminator
// CHECK:    }
// CHECK:    llvm.return
// CHECK:  }

// -----

// CHECK:  omp.declare_reduction @[[EQV_REDUCTION:.*]] : i32 init {
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
// CHECK-SAME: %[[ARRAY_REF:.*]]: !llvm.ptr
// CHECK:    %[[RED_ACCUMULATOR:.*]] = llvm.alloca %1 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr
// CHECK:    omp.parallel   {
// CHECK:      omp.wsloop reduction(@[[EQV_REDUCTION]] %[[RED_ACCUMULATOR]] -> %[[PRV:.+]] : !llvm.ptr) {
// CHECK-NEXT:   omp.loop_nest
// CHECK:          %[[ARRAY_ELEM_REF:.*]] = llvm.getelementptr %[[ARRAY_REF]][0, %{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK:          %[[ARRAY_ELEM:.*]] = llvm.load %[[ARRAY_ELEM_REF]] : !llvm.ptr -> i32
// CHECK:          %[[LPRV:.+]] = llvm.load %[[PRV]] : !llvm.ptr -> i32
// CHECK:          %[[ZERO_1:.*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK:          %[[ARGVAL_1:.*]] = llvm.icmp "ne" %[[LPRV]], %[[ZERO_1]] : i32
// CHECK:          %[[ZERO_2:.*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK:          %[[ARGVAL_2:.*]] = llvm.icmp "ne" %[[ARRAY_ELEM]], %[[ZERO_2]] : i32
// CHECK:          %[[RES:.*]] = llvm.icmp "eq" %[[ARGVAL_2]], %[[ARGVAL_1]] : i1
// CHECK:          %[[RES_EXT:.*]] = llvm.zext %[[RES]] : i1 to i32
// CHECK:          llvm.store %[[RES_EXT]], %[[PRV]] : i32, !llvm.ptr
// CHECK:          omp.yield
// CHECK:        omp.terminator
// CHECK:      omp.terminator
// CHECK:    llvm.return

omp.declare_reduction @eqv_reduction : !fir.logical<4> init {
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
    omp.wsloop reduction(@eqv_reduction %1 -> %prv : !fir.ref<!fir.logical<4>>) {
      omp.loop_nest (%arg1) : i32 = (%c1_i32) to (%c100_i32) inclusive step (%c1_i32_0) {
        fir.store %arg1 to %3 : !fir.ref<i32>
        %4 = fir.load %3 : !fir.ref<i32>
        %5 = fir.convert %4 : (i32) -> i64
        %c1_i64 = arith.constant 1 : i64
        %6 = arith.subi %5, %c1_i64 : i64
        %7 = fir.coordinate_of %arg0, %6 : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
        %8 = fir.load %7 : !fir.ref<!fir.logical<4>>
        %lprv = fir.load %prv : !fir.ref<!fir.logical<4>>
        %lprv1 = fir.convert %lprv : (!fir.logical<4>) -> i1
        %9 = fir.convert %8 : (!fir.logical<4>) -> i1
        %10 = arith.cmpi eq, %9, %lprv1 : i1
        %11 = fir.convert %10 : (i1) -> !fir.logical<4>
        fir.store %11 to %prv : !fir.ref<!fir.logical<4>>
        omp.yield
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

// CHECK: llvm.func @_QPs
// CHECK: omp.atomic.read %{{.*}} = %{{.*}}   : !llvm.ptr, !llvm.struct<(f32, f32)>

func.func @_QPs(%arg0: !fir.ref<!fir.complex<4>> {fir.bindc_name = "x"}) {
  %0 = fir.alloca !fir.complex<4> {bindc_name = "v", uniq_name = "_QFsEv"}
  omp.atomic.read %0 = %arg0   : !fir.ref<!fir.complex<4>>, !fir.complex<4>
  return
}

// -----

// Test if llvm.alloca is properly inserted in the omp section

//CHECK:  %[[CONST0:.*]] = llvm.mlir.constant(1 : i64) : i64
//CHECK:  %[[CONST:.*]] = llvm.mlir.constant(1 : i64) : i64
//CHECK:  %[[ALLOCA:.*]] = llvm.alloca %[[CONST]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {bindc_name = "iattr"} : (i64) -> !llvm.ptr
//CHECK:  omp.parallel   {
//CHECK:    %[[CONST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
//CHECK:    %[[ALLOCA_1:.*]] = llvm.alloca %[[CONST_1:.*]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
//CHECK:    %[[LOAD:.*]] = llvm.load %[[ALLOCA]] : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
//CHECK:    llvm.store %[[LOAD]], %[[ALLOCA_1]] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>, !llvm.ptr
//CHECK:    %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA_1]][0, 0] : (!llvm.ptr) -> !llvm.ptr
//CHECK:    %[[LOAD_2:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> !llvm.ptr
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
// CHECK: omp.ordered.region
  omp.ordered.region {
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

// -----

// CHECK-LABEL:  llvm.func @flush_standalone_
// CHECK-SAME:   %[[ARG_A:.*]]: !llvm.ptr {fir.bindc_name = "a"}, %[[ARG_B:.*]]: !llvm.ptr {fir.bindc_name = "b"}, %[[ARG_C:.*]]: !llvm.ptr {fir.bindc_name = "c"})
  func.func @flush_standalone_(%arg0: !fir.ref<i32> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "b"}, %arg2: !fir.ref<i32> {fir.bindc_name = "c"}) {
// CHECK:   omp.flush(%[[ARG_A]], %[[ARG_B]], %[[ARG_C]] : !llvm.ptr, !llvm.ptr, !llvm.ptr)
    omp.flush(%arg0, %arg1, %arg2 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
// CHECK:   omp.flush
    omp.flush
    return
  }

// CHECK-LABEL:  llvm.func @flush_parallel_
// CHECK-SAME:   %[[ARG_A:.*]]: !llvm.ptr {fir.bindc_name = "a"}, %[[ARG_B:.*]]: !llvm.ptr {fir.bindc_name = "b"}, %[[ARG_C:.*]]: !llvm.ptr {fir.bindc_name = "c"})
  func.func @flush_parallel_(%arg0: !fir.ref<i32> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "b"}, %arg2: !fir.ref<i32> {fir.bindc_name = "c"}) {
// CHECK:    omp.parallel   {
    omp.parallel   {
// CHECK:      omp.flush(%[[ARG_A]], %[[ARG_B]], %[[ARG_C]] : !llvm.ptr, !llvm.ptr, !llvm.ptr)
      omp.flush(%arg0, %arg1, %arg2 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
// CHECK:      omp.flush
      omp.flush
// CHECK:      %[[A_VAL:.*]] = llvm.load %[[ARG_A]] : !llvm.ptr -> i32
      %0 = fir.load %arg0 : !fir.ref<i32>
// CHECK:      %[[B_VAL:.*]] = llvm.load %[[ARG_B]] : !llvm.ptr -> i32
      %1 = fir.load %arg1 : !fir.ref<i32>
// CHECK:      %[[C_VAL:.*]] = llvm.add %[[A_VAL]], %[[B_VAL]]  : i32
      %2 = arith.addi %0, %1 : i32
// CHECK:      llvm.store %[[C_VAL]], %[[ARG_C]] : i32, !llvm.ptr
      fir.store %2 to %arg2 : !fir.ref<i32>
// CHECK:      omp.terminator
      omp.terminator
// CHECK:    }
    }
    return
  }

// -----

// CHECK:  omp.critical.declare @help   hint(contended)
omp.critical.declare @help hint(contended)

// CHECK: llvm.func @omp_critical_() {
func.func @omp_critical_() {
// CHECK: %[[Y_REF:.*]] = llvm.alloca %{{.*}} x i32 {bindc_name = "y"} : (i64) -> !llvm.ptr
  %0 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFomp_criticalEx"}
// CHECK: %[[X_REF:.*]] = llvm.alloca %{{.*}} x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr
  %1 = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFomp_criticalEy"}
// CHECK: omp.critical(@help)
  omp.critical(@help) {
// CHECK: %[[X_VAL:.*]] = llvm.load %[[X_REF]] : !llvm.ptr -> i32
    %2 = fir.load %0 : !fir.ref<i32>
// CHECK: %[[Y_VAL:.*]] = llvm.load %[[Y_REF]] : !llvm.ptr -> i32
    %3 = fir.load %1 : !fir.ref<i32>
// CHECK: %[[RESULT:.*]] = llvm.add %[[X_VAL]], %[[Y_VAL]]  : i32
    %4 = arith.addi %2, %3 : i32
// CHECK: llvm.store %[[RESULT]], %[[X_REF]] : i32, !llvm.ptr
    fir.store %4 to %0 : !fir.ref<i32>
// CHECK: omp.terminator
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL:  llvm.func @omp_map_info_descriptor_type_conversion
// CHECK-SAME:   %[[ARG_0:.*]]: !llvm.ptr)

func.func @omp_map_info_descriptor_type_conversion(%arg0 : !fir.ref<!fir.box<!fir.heap<i32>>>) {
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ARG_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
  %0 = fir.box_offset %arg0 base_addr : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
  // CHECK: %[[MEMBER_MAP:.*]] = omp.map.info var_ptr(%[[GEP]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
  %1 = omp.map.info var_ptr(%0 : !fir.llvm_ptr<!fir.ref<i32>>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
  // CHECK: %[[DESC_MAP:.*]] = omp.map.info var_ptr(%[[ARG_0]] : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(always, delete) capture(ByRef) members(%[[MEMBER_MAP]] : [0] : !llvm.ptr) -> !llvm.ptr {name = ""}
  %2 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(always, delete) capture(ByRef) members(%1 : [0] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.heap<i32>>> {name = ""}
  // CHECK: omp.target_exit_data map_entries(%[[DESC_MAP]] : !llvm.ptr) 
  omp.target_exit_data   map_entries(%2 : !fir.ref<!fir.box<!fir.heap<i32>>>)
  return 
}

// -----

// CHECK-LABEL:  llvm.func @omp_map_info_derived_type_explicit_member_conversion
// CHECK-SAME:   %[[ARG_0:.*]]: !llvm.ptr)

func.func @omp_map_info_derived_type_explicit_member_conversion(%arg0 : !fir.ref<!fir.type<_QFderived_type{real:f32,array:!fir.array<10xi32>,int:i32}>>) {
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ARG_0]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFderived_type", (f32, array<10 x i32>, i32)>
  %0 = fir.field_index int, !fir.type<_QFderived_type{real:f32,array:!fir.array<10xi32>,int:i32}>
  %1 = fir.coordinate_of %arg0, %0 : (!fir.ref<!fir.type<_QFderived_type{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.field) -> !fir.ref<i32>
  // CHECK: %[[MAP_MEMBER_1:.*]] = omp.map.info var_ptr(%[[GEP]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "dtype%int"}
  %2 = omp.map.info var_ptr(%1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "dtype%int"}
  // CHECK: %[[GEP_2:.*]] = llvm.getelementptr %[[ARG_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFderived_type", (f32, array<10 x i32>, i32)>
  %3 = fir.field_index real, !fir.type<_QFderived_type{real:f32,array:!fir.array<10xi32>,int:i32}>
  %4 = fir.coordinate_of %arg0, %3 : (!fir.ref<!fir.type<_QFderived_type{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.field) -> !fir.ref<f32>
  // CHECK: %[[MAP_MEMBER_2:.*]] = omp.map.info var_ptr(%[[GEP_2]] : !llvm.ptr, f32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "dtype%real"}
  %5 = omp.map.info var_ptr(%4 : !fir.ref<f32>, f32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f32> {name = "dtype%real"}    
  // CHECK: %[[MAP_PARENT:.*]] = omp.map.info var_ptr(%[[ARG_0]] : !llvm.ptr, !llvm.struct<"_QFderived_type", (f32, array<10 x i32>, i32)>) map_clauses(tofrom) capture(ByRef) members(%[[MAP_MEMBER_1]], %[[MAP_MEMBER_2]] : [2], [0] : !llvm.ptr, !llvm.ptr) -> !llvm.ptr {name = "dtype", partial_map = true} 
  %6 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.type<_QFderived_type{real:f32,array:!fir.array<10xi32>,int:i32}>>, !fir.type<_QFderived_type{real:f32,array:!fir.array<10xi32>,int:i32}>) map_clauses(tofrom) capture(ByRef) members(%2, %5 : [2], [0] : !fir.ref<i32>, !fir.ref<f32>) -> !fir.ref<!fir.type<_QFderived_type{real:f32,array:!fir.array<10xi32>,int:i32}>> {name = "dtype", partial_map = true}
  // CHECK: omp.target map_entries(%[[MAP_MEMBER_1]] -> %[[ARG_1:.*]], %[[MAP_MEMBER_2]] -> %[[ARG_2:.*]], %[[MAP_PARENT]] -> %[[ARG_3:.*]] : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
  // CHECK: ^bb0(%[[ARG_1]]: !llvm.ptr, %[[ARG_2]]: !llvm.ptr, %[[ARG_3]]: !llvm.ptr):
  omp.target map_entries(%2 -> %arg1, %5 -> %arg2, %6 -> %arg3 : !fir.ref<i32>, !fir.ref<f32>, !fir.ref<!fir.type<_QFderived_type{real:f32,array:!fir.array<10xi32>,int:i32}>>) {
  ^bb0(%arg1: !fir.ref<f32>, %arg2: !fir.ref<i32>, %arg3: !fir.ref<!fir.type<_QFderived_type{real:f32,array:!fir.array<10xi32>,int:i32}>>):
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL:  llvm.func @omp_map_info_nested_derived_type_explicit_member_conversion
// CHECK-SAME:   %[[ARG_0:.*]]: !llvm.ptr)

func.func @omp_map_info_nested_derived_type_explicit_member_conversion(%arg0 : !fir.ref<!fir.type<_QFTtop_layer{array_i:!fir.array<10xi32>,nested:!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>,k:i32}>>) {
    // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ARG_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFTtop_layer", (array<10 x i32>, struct<"_QFTbottom_layer", (array<10 x f32>, f64)>, i32)>
    %0 = fir.field_index nested, !fir.type<_QFTtop_layer{array_i:!fir.array<10xi32>,nested:!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>,k:i32}>
    %1 = fir.coordinate_of %arg0, %0 : (!fir.ref<!fir.type<_QFTtop_layer{array_i:!fir.array<10xi32>,nested:!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>,k:i32}>>, !fir.field) -> !fir.ref<!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>>
    // CHECK: %[[GEP_2:.*]] = llvm.getelementptr %[[GEP]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFTbottom_layer", (array<10 x f32>, f64)>
    %2 = fir.field_index i2, !fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>
    %3 = fir.coordinate_of %1, %2 : (!fir.ref<!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>>, !fir.field) -> !fir.ref<f64>
    // CHECK: %[[MAP_MEMBER_1:.*]] = omp.map.info var_ptr(%[[GEP_2]] : !llvm.ptr, f64) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %4 = omp.map.info var_ptr(%3 : !fir.ref<f64>, f64) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f64>
    // CHECK: %[[GEP_3:.*]] = llvm.getelementptr %[[ARG_0]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFTtop_layer", (array<10 x i32>, struct<"_QFTbottom_layer", (array<10 x f32>, f64)>, i32)>
    %5 = fir.field_index k, !fir.type<_QFTtop_layer{array_i:!fir.array<10xi32>,nested:!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>,k:i32}>
    %6 = fir.coordinate_of %arg0, %5 : (!fir.ref<!fir.type<_QFTtop_layer{array_i:!fir.array<10xi32>,nested:!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>,k:i32}>>, !fir.field) -> !fir.ref<i32>
    // CHECK: %[[MAP_MEMBER_2:.*]] = omp.map.info var_ptr(%[[GEP_3]] : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %7 = omp.map.info var_ptr(%6 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    // CHECK: %[[PARENT_MAP:.*]] = omp.map.info var_ptr(%[[ARG_0]] : !llvm.ptr, !llvm.struct<"_QFTtop_layer", (array<10 x i32>, struct<"_QFTbottom_layer", (array<10 x f32>, f64)>, i32)>) map_clauses(tofrom) capture(ByRef) members(%[[MAP_MEMBER_1]], %[[MAP_MEMBER_2]] : [1,1], [2,-1] : !llvm.ptr, !llvm.ptr) -> !llvm.ptr {partial_map = true}
    %9 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.type<_QFTtop_layer{array_i:!fir.array<10xi32>,nested:!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>,k:i32}>>, !fir.type<_QFTtop_layer{array_i:!fir.array<10xi32>,nested:!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>,k:i32}>) map_clauses(tofrom) capture(ByRef) members(%4, %7 : [1,1], [2,-1] : !fir.ref<f64>, !fir.ref<i32>) -> !fir.ref<!fir.type<_QFTtop_layer{array_i:!fir.array<10xi32>,nested:!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>,k:i32}>> {partial_map = true}
    // CHECK: omp.target map_entries(%[[MAP_MEMBER_1]] -> %{{.*}}, %[[MAP_MEMBER_2]] -> %{{.*}}, %[[PARENT_MAP]] -> %{{.*}} : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
    // CHECK: ^bb0(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr):
    omp.target map_entries(%4 -> %arg1, %7 -> %arg2, %9 -> %arg3 : !fir.ref<f64>, !fir.ref<i32>, !fir.ref<!fir.type<_QFTtop_layer{array_i:!fir.array<10xi32>,nested:!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>,k:i32}>>) {
     ^bb0(%arg1: !fir.ref<i32>, %arg2: !fir.ref<f64>, %arg3: !fir.ref<!fir.type<_QFTtop_layer{array_i:!fir.array<10xi32>,nested:!fir.type<_QFTbottom_layer{array_i2:!fir.array<10xf32>,i2:f64}>,k:i32}>>):
      omp.terminator
    }
  return
}

// -----
