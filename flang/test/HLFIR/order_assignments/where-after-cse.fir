// Test canonicalization of hlfir.exactly_once operations
// after CSE. The live-in values that are not dominating
// the top-level hlfir.where must be cloned inside hlfir.exactly_once,
// otherwise, the cloning of the hlfir.exactly_once before hlfir.where
// would cause def-use issues:
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

// Simple case, where CSE makes only hlfir.designate live-in:
// CHECK-LABEL:   func.func @_QPtest1(
func.func @_QPtest1(%arg0: !fir.ref<!fir.type<_QMmy_moduleTtt{p1:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>,p2:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>> {fir.bindc_name = "x"}) {
  %true = arith.constant true
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.type<_QMmy_moduleTtt{p1:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>,p2:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmy_moduleTtt{p1:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>,p2:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMmy_moduleTtt{p1:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>,p2:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>)
  hlfir.where {
    %2 = hlfir.designate %1#0{"p2"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmy_moduleTtt{p1:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>,p2:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %3 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    %4:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
    %5 = arith.addi %4#0, %4#1 : index
    %6 = arith.subi %5, %c1 : index
    %7 = arith.subi %6, %4#0 : index
    %8 = arith.addi %7, %c1 : index
    %9 = arith.cmpi sgt, %8, %c0 : index
    %10 = arith.select %9, %8, %c0 : index
    %11 = fir.shape %10 : (index) -> !fir.shape<1>
    %12 = hlfir.designate %3 (%4#0:%6:%c1)  shape %11 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
    %13 = hlfir.elemental %11 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
    ^bb0(%arg1: index):
      %14 = hlfir.designate %12 (%arg1)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      %15 = fir.load %14 : !fir.ref<f32>
      %16 = arith.cmpf ogt, %15, %cst fastmath<contract> : f32
      %17 = fir.convert %16 : (i1) -> !fir.logical<4>
      hlfir.yield_element %17 : !fir.logical<4>
    }
    hlfir.yield %13 : !hlfir.expr<?x!fir.logical<4>> cleanup {
      hlfir.destroy %13 : !hlfir.expr<?x!fir.logical<4>>
    }
  } do {
    hlfir.region_assign {
      %2 = hlfir.designate %1#0{"p1"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmy_moduleTtt{p1:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>,p2:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
      %3 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
      %4:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
      %5 = arith.addi %4#0, %4#1 : index
      %6 = arith.subi %5, %c1 : index
      %7 = arith.subi %6, %4#0 : index
      %8 = arith.addi %7, %c1 : index
      %9 = arith.cmpi sgt, %8, %c0 : index
      %10 = arith.select %9, %8, %c0 : index
      %11 = fir.shape %10 : (index) -> !fir.shape<1>
      %12 = hlfir.designate %3 (%4#0:%6:%c1, %c1)  shape %11 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
      %13 = hlfir.exactly_once : !hlfir.expr<?xf32> {
// CHECK:           %[[VAL_26:.*]] = hlfir.designate %{{.*}}#0{"p1"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmy_moduleTtt{p1:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>,p2:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
// CHECK:           fir.load %[[VAL_26]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
// CHECK:           %[[VAL_47:.*]] = fir.call @_QPcallee(%{{.*}}) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
// CHECK:           fir.do_loop
        %15 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
        %16:3 = fir.box_dims %15, %c0 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
        %17 = arith.addi %16#0, %16#1 : index
        %18 = arith.subi %17, %c1 : index
        %19 = arith.subi %18, %16#0 : index
        %20 = arith.addi %19, %c1 : index
        %21 = arith.cmpi sgt, %20, %c0 : index
        %22 = arith.select %21, %20, %c0 : index
        %23 = fir.shape %22 : (index) -> !fir.shape<1>
        %24 = hlfir.designate %15 (%16#0:%18:%c1, %c1)  shape %23 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
        %25:2 = hlfir.declare %24 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMmy_moduleFcalleeEx"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
        %26:3 = fir.box_dims %25#0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
        %27 = fir.convert %26#1 : (index) -> i64
        %28 = fir.convert %27 : (i64) -> index
        %29 = arith.cmpi sgt, %28, %c0 : index
        %30 = arith.select %29, %28, %c0 : index
        %31 = fir.shape %30 : (index) -> !fir.shape<1>
        %32 = fir.allocmem !fir.array<?xf32>, %30 {bindc_name = ".tmp.expr_result", uniq_name = ""}
        %33 = fir.convert %32 : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
        %34:2 = hlfir.declare %33(%31) {uniq_name = ".tmp.expr_result"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
        %35 = fir.call @_QPcallee(%24) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
        fir.save_result %35 to %34#1(%31) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
        %36 = hlfir.as_expr %34#0 move %true : (!fir.box<!fir.array<?xf32>>, i1) -> !hlfir.expr<?xf32>
        hlfir.yield %36 : !hlfir.expr<?xf32> cleanup {
          hlfir.destroy %36 : !hlfir.expr<?xf32>
        }
      }
      %14 = hlfir.elemental %11 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
      ^bb0(%arg1: index):
        %15 = hlfir.designate %12 (%arg1)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
        %16 = hlfir.apply %13, %arg1 : (!hlfir.expr<?xf32>, index) -> f32
        %17 = fir.load %15 : !fir.ref<f32>
        %18 = arith.divf %17, %16 fastmath<contract> : f32
        hlfir.yield_element %18 : f32
      }
      hlfir.yield %14 : !hlfir.expr<?xf32> cleanup {
        hlfir.destroy %14 : !hlfir.expr<?xf32>
      }
    } to {
      %2 = hlfir.designate %1#0{"p2"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmy_moduleTtt{p1:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>,p2:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
      %3 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
      %4:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
      %5 = arith.addi %4#0, %4#1 : index
      %6 = arith.subi %5, %c1 : index
      %7 = arith.subi %6, %4#0 : index
      %8 = arith.addi %7, %c1 : index
      %9 = arith.cmpi sgt, %8, %c0 : index
      %10 = arith.select %9, %8, %c0 : index
      %11 = fir.shape %10 : (index) -> !fir.shape<1>
      %12 = hlfir.designate %3 (%4#0:%6:%c1)  shape %11 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
      hlfir.yield %12 : !fir.box<!fir.array<?xf32>> 
    }
  }
  return
}

// CSE makes a chain of operations live-in:
// CHECK-LABEL:   func.func @_QPtest_where_in_forall(
func.func @_QPtest_where_in_forall(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "b"}, %arg2: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "c"}) {
  %false = arith.constant false
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %c100 = arith.constant 100 : index
  %0 = fir.alloca !fir.array<100x!fir.logical<4>> {bindc_name = ".tmp.expr_result"}
  %1 = fir.alloca !fir.array<100x!fir.logical<4>> {bindc_name = ".tmp.expr_result"}
  %2 = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_21:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} {uniq_name = "_QFtest_where_in_forallEb"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %2 {uniq_name = "_QFtest_where_in_forallEa"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  %4:2 = hlfir.declare %arg1 dummy_scope %2 {uniq_name = "_QFtest_where_in_forallEb"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  %5:2 = hlfir.declare %arg2 dummy_scope %2 {uniq_name = "_QFtest_where_in_forallEc"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  hlfir.forall lb {
    hlfir.yield %c1_i32 : i32 
  } ub {
    hlfir.yield %c10_i32 : i32 
  }  (%arg3: i32) {
    hlfir.where {
      %6 = fir.shape %c100 : (index) -> !fir.shape<1>
      %7:2 = hlfir.declare %0(%6) {uniq_name = ".tmp.expr_result"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
      %8 = fir.call @_QPpure_logical_func1() proc_attrs<pure> fastmath<contract> : () -> !fir.array<100x!fir.logical<4>>
      fir.save_result %8 to %7#1(%6) : !fir.array<100x!fir.logical<4>>, !fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>
      %9 = hlfir.as_expr %7#0 move %false : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i1) -> !hlfir.expr<100x!fir.logical<4>>
      hlfir.yield %9 : !hlfir.expr<100x!fir.logical<4>> cleanup {
        hlfir.destroy %9 : !hlfir.expr<100x!fir.logical<4>>
      }
    } do {
      hlfir.region_assign {
        %6 = fir.convert %arg3 : (i32) -> i64
// CHECK:             %[[VAL_58:.*]]:3 = fir.box_dims %[[VAL_21]]#1, %{{.*}} : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_59:.*]] = arith.cmpi sgt, %[[VAL_58]]#1, %{{.*}} : index
// CHECK:             %[[VAL_60:.*]] = arith.select %[[VAL_59]], %[[VAL_58]]#1, %{{.*}} : index
// CHECK:             %[[VAL_61:.*]] = fir.shape %[[VAL_60]] : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_62:.*]] = hlfir.designate %[[VAL_21]]#0 (%{{.*}}, %{{.*}}:%[[VAL_58]]#1:%{{.*}})  shape %[[VAL_61]] : (!fir.box<!fir.array<?x?xf32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
        %7:3 = fir.box_dims %4#1, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
        %8 = arith.cmpi sgt, %7#1, %c0 : index
        %9 = arith.select %8, %7#1, %c0 : index
        %10 = fir.shape %9 : (index) -> !fir.shape<1>
        %11 = hlfir.designate %4#0 (%6, %c1:%7#1:%c1)  shape %10 : (!fir.box<!fir.array<?x?xf32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
        %12 = hlfir.exactly_once : f32 {
          %19:3 = fir.box_dims %3#1, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
          %20 = arith.cmpi sgt, %19#1, %c0 : index
          %21 = arith.select %20, %19#1, %c0 : index
          %22 = fir.shape %21 : (index) -> !fir.shape<1>
          %23 = hlfir.designate %3#0 (%6, %c1:%19#1:%c1)  shape %22 : (!fir.box<!fir.array<?x?xf32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
// CHECK:             %[[VAL_68:.*]] = fir.call @_QPpure_real_func2() fastmath<contract> : () -> f32
// CHECK:             %[[VAL_69:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:             ^bb0(%[[VAL_70:.*]]: index):
// CHECK:               %[[VAL_72:.*]] = hlfir.designate %[[VAL_62]] (%[[VAL_70]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
          %24 = fir.call @_QPpure_real_func2() fastmath<contract> : () -> f32
          %25 = hlfir.elemental %22 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
          ^bb0(%arg4: index):
            %28 = hlfir.designate %23 (%arg4)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
            %29 = hlfir.designate %11 (%arg4)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
            %30 = fir.load %28 : !fir.ref<f32>
            %31 = fir.load %29 : !fir.ref<f32>
            %32 = arith.addf %30, %31 fastmath<contract> : f32
            %33 = arith.addf %32, %24 fastmath<contract> : f32
            hlfir.yield_element %33 : f32
          }
          %26:3 = hlfir.associate %25(%22) {adapt.valuebyref} : (!hlfir.expr<?xf32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, i1)
          %27 = fir.call @_QPpure_real_func(%26#1) fastmath<contract> : (!fir.ref<!fir.array<?xf32>>) -> f32
          hlfir.yield %27 : f32 cleanup {
            hlfir.end_associate %26#1, %26#2 : !fir.ref<!fir.array<?xf32>>, i1
            hlfir.destroy %25 : !hlfir.expr<?xf32>
          }
        }
        %13:3 = fir.box_dims %3#1, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
        %14 = arith.cmpi sgt, %13#1, %c0 : index
        %15 = arith.select %14, %13#1, %c0 : index
        %16 = fir.shape %15 : (index) -> !fir.shape<1>
        %17 = hlfir.designate %3#0 (%6, %c1:%13#1:%c1)  shape %16 : (!fir.box<!fir.array<?x?xf32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
        %18 = hlfir.elemental %10 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
        ^bb0(%arg4: index):
          %19 = hlfir.designate %11 (%arg4)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
          %20 = fir.load %19 : !fir.ref<f32>
          %21 = arith.addf %20, %12 fastmath<contract> : f32
          %22 = hlfir.designate %17 (%arg4)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
          %23 = fir.call @_QPpure_elem_func(%22) proc_attrs<elemental, pure> fastmath<contract> : (!fir.ref<f32>) -> f32
          %24 = arith.addf %21, %23 fastmath<contract> : f32
          hlfir.yield_element %24 : f32
        }
        hlfir.yield %18 : !hlfir.expr<?xf32> cleanup {
          hlfir.destroy %18 : !hlfir.expr<?xf32>
        }
      } to {
        %6 = arith.muli %arg3, %c2_i32 overflow<nsw> : i32
        %7 = fir.convert %6 : (i32) -> i64
        %8:3 = fir.box_dims %3#1, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
        %9 = arith.cmpi sgt, %8#1, %c0 : index
        %10 = arith.select %9, %8#1, %c0 : index
        %11 = fir.shape %10 : (index) -> !fir.shape<1>
        %12 = hlfir.designate %3#0 (%7, %c1:%8#1:%c1)  shape %11 : (!fir.box<!fir.array<?x?xf32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
        hlfir.yield %12 : !fir.box<!fir.array<?xf32>> 
      }
      hlfir.elsewhere mask {
        %6 = hlfir.exactly_once : !hlfir.expr<100x!fir.logical<4>> {
          %7 = fir.shape %c100 : (index) -> !fir.shape<1>
          %8:2 = hlfir.declare %1(%7) {uniq_name = ".tmp.expr_result"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
          %9 = fir.call @_QPpure_logical_func2() proc_attrs<pure> fastmath<contract> : () -> !fir.array<100x!fir.logical<4>>
          fir.save_result %9 to %8#1(%7) : !fir.array<100x!fir.logical<4>>, !fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>
          %10 = hlfir.as_expr %8#0 move %false : (!fir.ref<!fir.array<100x!fir.logical<4>>>, i1) -> !hlfir.expr<100x!fir.logical<4>>
          hlfir.yield %10 : !hlfir.expr<100x!fir.logical<4>> cleanup {
            hlfir.destroy %10 : !hlfir.expr<100x!fir.logical<4>>
          }
        }
        hlfir.yield %6 : !hlfir.expr<100x!fir.logical<4>> 
      } do {
        hlfir.region_assign {
          %6 = fir.convert %arg3 : (i32) -> i64
          %7:3 = fir.box_dims %5#1, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
          %8 = arith.cmpi sgt, %7#1, %c0 : index
          %9 = arith.select %8, %7#1, %c0 : index
          %10 = fir.shape %9 : (index) -> !fir.shape<1>
          %11 = hlfir.designate %5#0 (%6, %c1:%7#1:%c1)  shape %10 : (!fir.box<!fir.array<?x?xf32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
          hlfir.yield %11 : !fir.box<!fir.array<?xf32>> 
        } to {
          %6 = arith.muli %arg3, %c2_i32 overflow<nsw> : i32
          %7 = fir.convert %6 : (i32) -> i64
          %8 = hlfir.exactly_once : i32 {
            %14 = fir.call @_QPpure_ifoo() proc_attrs<pure> fastmath<contract> : () -> i32
            hlfir.yield %14 : i32 cleanup {
            }
          }
          %9 = fir.convert %8 : (i32) -> index
          %10 = arith.cmpi sgt, %9, %c0 : index
          %11 = arith.select %10, %9, %c0 : index
          %12 = fir.shape %11 : (index) -> !fir.shape<1>
          %13 = hlfir.designate %3#0 (%7, %c1:%9:%c1)  shape %12 : (!fir.box<!fir.array<?x?xf32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
          hlfir.yield %13 : !fir.box<!fir.array<?xf32>> 
        }
      }
    }
  }
  return
}
