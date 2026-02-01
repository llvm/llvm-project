// Test inlining of hlfir.copy_in and hlfir.copy_out
// RUN: fir-opt --inline-hlfir-copy %s | FileCheck %s

// Test inlining of hlfir.copy_in that does not require the array to be copied out
func.func private @_test_inline_copy_in(%arg0: !fir.box<!fir.array<?x?x?xf64>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "i"}, %arg2: !fir.ref<i32> {fir.bindc_name = "j"}) {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf64>>>
  %1 = fir.dummy_scope : !fir.dscope
  %2:2 = hlfir.declare %arg1 dummy_scope %1 {uniq_name = "_QFFsb2Ei"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg2 dummy_scope %1 {uniq_name = "_QFFsb2Ej"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg0 dummy_scope %1 {uniq_name = "_QFFsb2Ex"} : (!fir.box<!fir.array<?x?x?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xf64>>, !fir.box<!fir.array<?x?x?xf64>>)
  %5 = fir.load %2#0 : !fir.ref<i32>
  %6 = fir.convert %5 : (i32) -> i64
  %c1 = arith.constant 1 : index
  %c1_0 = arith.constant 1 : index
  %7:3 = fir.box_dims %4#1, %c1_0 : (!fir.box<!fir.array<?x?x?xf64>>, index) -> (index, index, index)
  %c1_1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %8 = arith.subi %7#1, %c1 : index
  %9 = arith.addi %8, %c1_1 : index
  %10 = arith.divsi %9, %c1_1 : index
  %11 = arith.cmpi sgt, %10, %c0 : index
  %12 = arith.select %11, %10, %c0 : index
  %13 = fir.load %3#0 : !fir.ref<i32>
  %14 = fir.convert %13 : (i32) -> i64
  %15 = fir.shape %12 : (index) -> !fir.shape<1>
  %16 = hlfir.designate %4#0 (%6, %c1:%7#1:%c1_1, %14)  shape %15 : (!fir.box<!fir.array<?x?x?xf64>>, i64, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
  %c100_i32 = arith.constant 100 : i32
  %17:2 = hlfir.copy_in %16 to %0 : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.box<!fir.array<?xf64>>, i1)
  %18 = fir.box_addr %17#0 : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
  %19:3 = hlfir.associate %c100_i32 {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  fir.call @_QFPsb(%18, %19#0) fastmath<contract> : (!fir.ref<!fir.array<?xf64>>, !fir.ref<i32>) -> ()
  hlfir.copy_out %0, %17#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1) -> ()
  hlfir.end_associate %19#1, %19#2 : !fir.ref<i32>, i1
  return
}

// CHECK: #loop_vectorize = #llvm.loop_vectorize<disable = true>
// CHECK: #loop_annotation = #llvm.loop_annotation<vectorize = #loop_vectorize>
// CHECK-LABEL:   func.func private @_test_inline_copy_in(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x?xf64>> {fir.bindc_name = "x"},
// CHECK-SAME:                                          %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
// CHECK-SAME:                                          %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "j"}) {
// CHECK:    %[[VAL_3:.*]] = arith.constant true
// CHECK:    %[[VAL_4:.*]] = arith.constant false
// CHECK:    %[[VAL_5:.*]] = arith.constant 100 : i32
// CHECK:    %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:    %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:    %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:    %[[VAL_22:.*]]:2 = hlfir.declare %[[VAL_1:.*]] dummy_scope %[[VAL_8:.*]] {uniq_name = "_QFFsb2Ei"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:    %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2:.*]] dummy_scope %[[VAL_8:.*]] {uniq_name = "_QFFsb2Ej"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:    %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0:.*]] dummy_scope %[[VAL_8:.*]] {uniq_name = "_QFFsb2Ex"} : (!fir.box<!fir.array<?x?x?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xf64>>, !fir.box<!fir.array<?x?x?xf64>>)
// CHECK:    %[[VAL_11:.*]] = fir.load %[[VAL_22:.*]]#0 : !fir.ref<i32>
// CHECK:    %[[VAL_12:.*]] = fir.convert %[[VAL_11:.*]] : (i32) -> i64
// CHECK:    %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_10:.*]]#1, %[[VAL_7:.*]] : (!fir.box<!fir.array<?x?x?xf64>>, index) -> (index, index, index)
// CHECK:    %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_13:.*]]#1, %[[VAL_6:.*]] : index
// CHECK:    %[[VAL_15:.*]] = arith.select %[[VAL_14:.*]], %[[VAL_13:.*]]#1, %[[VAL_6:.*]] : index
// CHECK:    %[[VAL_16:.*]] = fir.load %[[VAL_9:.*]]#0 : !fir.ref<i32>
// CHECK:    %[[VAL_17:.*]] = fir.convert %[[VAL_16:.*]] : (i32) -> i64
// CHECK:    %[[VAL_18:.*]] = fir.shape %[[VAL_15:.*]] : (index) -> !fir.shape<1>
// CHECK:    %[[VAL_19:.*]] = hlfir.designate %[[VAL_10:.*]]#0 (%[[VAL_12:.*]], %[[VAL_7:.*]]:%[[VAL_13:.*]]#1:%[[VAL_7:.*]], %[[VAL_17:.*]])  shape %[[VAL_18:.*]] : (!fir.box<!fir.array<?x?x?xf64>>, i64, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
// CHECK:    %[[VAL_20:.*]] = fir.is_contiguous_box %[[VAL_19:.*]] whole : (!fir.box<!fir.array<?xf64>>) -> i1
// CHECK:    %[[VAL_21:.*]]:2 = fir.if %[[VAL_20:.*]] -> (!fir.box<!fir.array<?xf64>>, i1) {
// CHECK:      fir.result %[[VAL_19:.*]], %[[VAL_4:.*]] : !fir.box<!fir.array<?xf64>>, i1
// CHECK:    } else {
// CHECK:      %[[VAL_24:.*]] = fir.allocmem !fir.array<?xf64>, %[[VAL_15:.*]] {bindc_name = ".tmp.copy_in", uniq_name = ""}
// CHECK:      %[[VAL_25:.*]]:2 = hlfir.declare %[[VAL_24:.*]](%[[VAL_18:.*]]) {uniq_name = ".tmp.copy_in"} : (!fir.heap<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.heap<!fir.array<?xf64>>)
// CHECK:      fir.do_loop %arg3 = %[[VAL_7:.*]] to %[[VAL_15:.*]] step %[[VAL_7:.*]] unordered attributes {loopAnnotation = #loop_annotation} {
// CHECK:        %[[VAL_26:.*]] = hlfir.designate %[[VAL_19:.*]] (%arg3)  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:        %[[VAL_27:.*]] = fir.load %[[VAL_26:.*]] : !fir.ref<f64>
// CHECK:        %[[VAL_28:.*]] = hlfir.designate %[[VAL_25:.*]]#0 (%arg3)  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:        hlfir.assign %[[VAL_27:.*]] to %[[VAL_28:.*]] : f64, !fir.ref<f64>
// CHECK:      }
// CHECK:      fir.result %[[VAL_25:.*]]#0, %[[VAL_3:.*]] : !fir.box<!fir.array<?xf64>>, i1
// CHECK:    }
// CHECK:    %[[VAL_ALLOCA:.*]] = fir.alloca !fir.box<!fir.array<?xf64>>
// CHECK:    fir.store %[[VAL_21:.*]]#0 to %[[VAL_ALLOCA:.*]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
// CHECK:    %[[VAL_22:.*]] = fir.box_addr %[[VAL_21:.*]]#0 : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
// CHECK:    %[[VAL_23:.*]]:3 = hlfir.associate %[[VAL_5:.*]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
// CHECK:    fir.call @_QFPsb(%[[VAL_22:.*]], %[[VAL_23:.*]]#0) fastmath<contract> : (!fir.ref<!fir.array<?xf64>>, !fir.ref<i32>) -> ()
// CHECK:    fir.if %[[VAL_21:.*]]#1 {
// CHECK:      %[[VAL_BOX:.*]] = fir.load %[[VAL_ALLOCA:.*]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
// CHECK:      %[[VAL_ADDR:.*]] = fir.box_addr %[[VAL_BOX:.*]] : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
// CHECK:      %[[VAL_HEAP:.*]] = fir.convert %[[VAL_ADDR:.*]] : (!fir.ref<!fir.array<?xf64>>) -> !fir.heap<!fir.array<?xf64>>
// CHECK:      fir.freemem %[[VAL_HEAP:.*]] : !fir.heap<!fir.array<?xf64>>
// CHECK:    }
// CHECK:    hlfir.end_associate %[[VAL_23:.*]]#1, %[[VAL_23:.*]]#2 : !fir.ref<i32>, i1
// CHECK:    return
// CHECK:  }

// Test inlining of hlfir.copy_in with copy-back (intent(out/inout))
func.func private @_test_inline_copy_in_with_copyback(%arg0: !fir.box<!fir.array<?x?x?xf64>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "i"}, %arg2: !fir.ref<i32> {fir.bindc_name = "j"}) {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf64>>>
  %1 = fir.dummy_scope : !fir.dscope
  %2:2 = hlfir.declare %arg1 dummy_scope %1 {uniq_name = "_QFFsb2Ei"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg2 dummy_scope %1 {uniq_name = "_QFFsb2Ej"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg0 dummy_scope %1 {uniq_name = "_QFFsb2Ex"} : (!fir.box<!fir.array<?x?x?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xf64>>, !fir.box<!fir.array<?x?x?xf64>>)
  %5 = fir.load %2#0 : !fir.ref<i32>
  %6 = fir.convert %5 : (i32) -> i64
  %c1 = arith.constant 1 : index
  %c1_0 = arith.constant 1 : index
  %7:3 = fir.box_dims %4#1, %c1_0 : (!fir.box<!fir.array<?x?x?xf64>>, index) -> (index, index, index)
  %c1_1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %8 = arith.subi %7#1, %c1 : index
  %9 = arith.addi %8, %c1_1 : index
  %10 = arith.divsi %9, %c1_1 : index
  %11 = arith.cmpi sgt, %10, %c0 : index
  %12 = arith.select %11, %10, %c0 : index
  %13 = fir.load %3#0 : !fir.ref<i32>
  %14 = fir.convert %13 : (i32) -> i64
  %15 = fir.shape %12 : (index) -> !fir.shape<1>
  %16 = hlfir.designate %4#0 (%6, %c1:%7#1:%c1_1, %14)  shape %15 : (!fir.box<!fir.array<?x?x?xf64>>, i64, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
  %c100_i32 = arith.constant 100 : i32
  %17:2 = hlfir.copy_in %16 to %0 : (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> (!fir.box<!fir.array<?xf64>>, i1)
  %18 = fir.box_addr %17#0 : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
  %19:3 = hlfir.associate %c100_i32 {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  fir.call @_QFPsb(%18, %19#1) fastmath<contract> : (!fir.ref<!fir.array<?xf64>>, !fir.ref<i32>) -> ()
  hlfir.copy_out %0, %17#1 to %16 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>, i1, !fir.box<!fir.array<?xf64>>) -> ()
  hlfir.end_associate %19#1, %19#2 : !fir.ref<i32>, i1
  return
}

// CHECK-LABEL:  func.func private @_test_inline_copy_in_with_copyback(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x?xf64>> {fir.bindc_name = "x"},
// CHECK-SAME:                                             %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
// CHECK-SAME:                                             %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "j"}) {
// CHECK:    %[[VAL_TRUE:.*]] = arith.constant true
// CHECK:    %[[VAL_FALSE:.*]] = arith.constant false
// CHECK:    %[[VAL_100:.*]] = arith.constant 100 : i32
// CHECK:    %[[VAL_C0:.*]] = arith.constant 0 : index
// CHECK:    %[[VAL_C1:.*]] = arith.constant 1 : index
// CHECK:    %[[VAL_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:    %[[VAL_I:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_SCOPE]] {uniq_name = "_QFFsb2Ei"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:    %[[VAL_J:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_SCOPE]] {uniq_name = "_QFFsb2Ej"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:    %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_SCOPE]] {uniq_name = "_QFFsb2Ex"} : (!fir.box<!fir.array<?x?x?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xf64>>, !fir.box<!fir.array<?x?x?xf64>>)
// CHECK:    %[[VAL_ILOAD:.*]] = fir.load %[[VAL_I]]#0 : !fir.ref<i32>
// CHECK:    %[[VAL_ICVT:.*]] = fir.convert %[[VAL_ILOAD]] : (i32) -> i64
// CHECK:    %[[VAL_DIMS:.*]]:3 = fir.box_dims %[[VAL_X]]#1, %[[VAL_C1]] : (!fir.box<!fir.array<?x?x?xf64>>, index) -> (index, index, index)
// CHECK:    %[[VAL_CMP:.*]] = arith.cmpi sgt, %[[VAL_DIMS]]#1, %[[VAL_C0]] : index
// CHECK:    %[[VAL_EXTENT:.*]] = arith.select %[[VAL_CMP]], %[[VAL_DIMS]]#1, %[[VAL_C0]] : index
// CHECK:    %[[VAL_JLOAD:.*]] = fir.load %[[VAL_J]]#0 : !fir.ref<i32>
// CHECK:    %[[VAL_JCVT:.*]] = fir.convert %[[VAL_JLOAD]] : (i32) -> i64
// CHECK:    %[[VAL_SHAPE:.*]] = fir.shape %[[VAL_EXTENT]] : (index) -> !fir.shape<1>
// CHECK:    %[[VAL_DESIGNATE:.*]] = hlfir.designate %[[VAL_X]]#0 (%[[VAL_ICVT]], %[[VAL_C1]]:%[[VAL_DIMS]]#1:%[[VAL_C1]], %[[VAL_JCVT]])  shape %[[VAL_SHAPE]] : (!fir.box<!fir.array<?x?x?xf64>>, i64, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
// CHECK:    %[[VAL_ISCONTIG:.*]] = fir.is_contiguous_box %[[VAL_DESIGNATE]] whole : (!fir.box<!fir.array<?xf64>>) -> i1
// CHECK:    %[[VAL_IFRES:.*]]:2 = fir.if %[[VAL_ISCONTIG]] -> (!fir.box<!fir.array<?xf64>>, i1) {
// CHECK:      fir.result %[[VAL_DESIGNATE]], %[[VAL_FALSE]] : !fir.box<!fir.array<?xf64>>, i1
// CHECK:    } else {
// CHECK:      %[[VAL_ALLOC:.*]] = fir.allocmem !fir.array<?xf64>, %[[VAL_EXTENT]] {bindc_name = ".tmp.copy_in", uniq_name = ""}
// CHECK:      %[[VAL_DECL:.*]]:2 = hlfir.declare %[[VAL_ALLOC]](%[[VAL_SHAPE]]) {uniq_name = ".tmp.copy_in"} : (!fir.heap<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.heap<!fir.array<?xf64>>)
// CHECK:      fir.do_loop %[[ARG:.*]] = %[[VAL_C1]] to %[[VAL_EXTENT]] step %[[VAL_C1]] unordered {{.*}} {
// CHECK:        %[[VAL_SRC:.*]] = hlfir.designate %[[VAL_DESIGNATE]] (%[[ARG]])  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:        %[[VAL_SRCVAL:.*]] = fir.load %[[VAL_SRC]] : !fir.ref<f64>
// CHECK:        %[[VAL_DST:.*]] = hlfir.designate %[[VAL_DECL]]#0 (%[[ARG]])  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:        hlfir.assign %[[VAL_SRCVAL]] to %[[VAL_DST]] : f64, !fir.ref<f64>
// CHECK:      }
// CHECK:      fir.result %[[VAL_DECL]]#0, %[[VAL_TRUE]] : !fir.box<!fir.array<?xf64>>, i1
// CHECK:    }
// CHECK:    %[[VAL_ALLOCA:.*]] = fir.alloca !fir.box<!fir.array<?xf64>>
// CHECK:    fir.store %[[VAL_IFRES]]#0 to %[[VAL_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
// CHECK:    %[[VAL_BOXADDR:.*]] = fir.box_addr %[[VAL_IFRES]]#0 : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
// CHECK:    %[[VAL_ASSOC:.*]]:3 = hlfir.associate %[[VAL_100]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
// CHECK:    fir.call @_QFPsb(%[[VAL_BOXADDR]], %[[VAL_ASSOC]]#1) fastmath<contract> : (!fir.ref<!fir.array<?xf64>>, !fir.ref<i32>) -> ()
// CHECK:    fir.if %[[VAL_IFRES]]#1 {
// CHECK:      %[[VAL_LOADBOX:.*]] = fir.load %[[VAL_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xf64>>>
// CHECK:      fir.do_loop %[[ARG2:.*]] = %[[VAL_C1]] to %[[VAL_EXTENT]] step %[[VAL_C1]] unordered {{.*}} {
// CHECK:        fir.box_dims %[[VAL_LOADBOX]]
// CHECK:        hlfir.designate %[[VAL_LOADBOX]]
// CHECK:        fir.load
// CHECK:        hlfir.designate %[[VAL_DESIGNATE]] (%[[ARG2]])  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:        hlfir.assign {{.*}} : f64, !fir.ref<f64>
// CHECK:      }
// CHECK:      %[[VAL_FREEADDR:.*]] = fir.box_addr %[[VAL_LOADBOX]] : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
// CHECK:      %[[VAL_FREEHEAP:.*]] = fir.convert %[[VAL_FREEADDR]] : (!fir.ref<!fir.array<?xf64>>) -> !fir.heap<!fir.array<?xf64>>
// CHECK:      fir.freemem %[[VAL_FREEHEAP]] : !fir.heap<!fir.array<?xf64>>
// CHECK:    }
// CHECK:    hlfir.end_associate %[[VAL_ASSOC]]#1, %[[VAL_ASSOC]]#2 : !fir.ref<i32>, i1
// CHECK:    return
// CHECK:  }

// Test not inlining optional dummy arguments (no direct copy-out)
func.func @_QPoptional_copy_in_out(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
  %false = arith.constant false
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
  %1 = fir.dummy_scope : !fir.dscope
  %2:2 = hlfir.declare %arg0 dummy_scope %1 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFoptional_copy_in_outEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %3 = fir.is_present %2#0 : (!fir.box<!fir.array<?xf32>>) -> i1
  %4:2 = fir.if %3 -> (!fir.ref<!fir.array<?xf32>>, i1) {
    %5:2 = hlfir.copy_in %2#0 to %0 : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
    %6 = fir.box_addr %5#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
    fir.result %6, %5#1 : !fir.ref<!fir.array<?xf32>>, i1
  } else {
    %5 = fir.absent !fir.ref<!fir.array<?xf32>>
    fir.result %5, %false : !fir.ref<!fir.array<?xf32>>, i1
  }
  fir.call @_QPtakes_optional_explicit(%4#0) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
  hlfir.copy_out %0, %4#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1) -> ()
  return
}

// CHECK-LABEL:  func.func @_QPoptional_copy_in_out(
// CHECK-SAME: %[[ARG_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
// CHECK:    %false = arith.constant false
// CHECK:    %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:    %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:    %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG_0:.*]] dummy_scope %[[VAL_1:.*]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFoptional_copy_in_outEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:    %[[VAL_3:.*]] = fir.is_present %[[VAL_2:.*]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
// CHECK:    %[[VAL_4:.*]]:2 = fir.if %[[VAL_3:.*]] -> (!fir.ref<!fir.array<?xf32>>, i1) {
// CHECK:      %[[VAL_5:.*]]:2 = hlfir.copy_in %[[VAL_2:.*]]#0 to %[[VAL_0:.*]] : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.box<!fir.array<?xf32>>, i1)
// CHECK:      %[[VAL_6:.*]] = fir.box_addr %[[VAL_5:.*]]#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:      fir.result %[[VAL_6:.*]], %[[VAL_5:.*]]#1 : !fir.ref<!fir.array<?xf32>>, i1
// CHECK:    } else {
// CHECK:      %[[VAL_5:.*]] = fir.absent !fir.ref<!fir.array<?xf32>>
// CHECK:      fir.result %[[VAL_5:.*]], %false : !fir.ref<!fir.array<?xf32>>, i1
// CHECK:    }
// CHECK:    fir.call @_QPtakes_optional_explicit(%[[VAL_4:.*]]#0) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> ()
// CHECK:    hlfir.copy_out %[[VAL_0:.*]], %[[VAL_4:.*]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i1) -> ()
// CHECK:    return
// CHECK:  }

// Test not inlining of assumed-rank arrays
func.func @_QPtest_copy_in_out_2(%arg0: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<*:f32>>>
  %1 = fir.dummy_scope : !fir.dscope
  %2:2 = hlfir.declare %arg0 dummy_scope %1 {uniq_name = "_QFtest_copy_in_out_2Ex"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
  %3:2 = hlfir.copy_in %2#0 to %0 : (!fir.box<!fir.array<*:f32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> (!fir.box<!fir.array<*:f32>>, i1)
  fir.call @_QPtakes_contiguous_intentin(%3#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
  hlfir.copy_out %0, %3#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, i1) -> ()
  return
}

// CHECK-LABEL: func.func @_QPtest_copy_in_out_2(
// CHECK-SAME: %[[ARG_0]]: !fir.box<!fir.array<*:f32>> {fir.bindc_name = "x"}) {
// CHECK:   %[[VAL_0]] = fir.alloca !fir.box<!fir.heap<!fir.array<*:f32>>>
// CHECK:   %[[VAL_1]] = fir.dummy_scope : !fir.dscope
// CHECK:   %[[VAL_2]]:2 = hlfir.declare %[[ARG_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_copy_in_out_2Ex"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
// CHECK:   %[[VAL_3]]:2 = hlfir.copy_in %[[VAL_2]]#0 to %[[VAL_0]] : (!fir.box<!fir.array<*:f32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>) -> (!fir.box<!fir.array<*:f32>>, i1)
// CHECK:   fir.call @_QPtakes_contiguous_intentin(%[[VAL_3]]#0) fastmath<contract> : (!fir.box<!fir.array<*:f32>>) -> ()
// CHECK:   hlfir.copy_out %[[VAL_0]], %[[VAL_3]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<*:f32>>>>, i1) -> ()
// CHECK:   return
// CHECK: }
