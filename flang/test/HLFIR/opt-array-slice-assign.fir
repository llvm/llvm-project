// Test optimized bufferization for hlfir.assign of array
// slices, e.g.:
//   x(2:7999,1:120,new) = (x(2:7999,1:120,old))
// We can expand hlfir.assign if the slices are either identical
// or completely disjoint. In case they are identical, we still
// need to make sure that the one-based indices are used
// uniformly for both LHS and RHS.
// RUN: fir-opt --opt-bufferization %s | FileCheck %s

func.func @_QPtest1(%arg0: !fir.ref<!fir.array<8000x120x3xf32>> {fir.bindc_name = "x"}) {
  %c7998 = arith.constant 7998 : index
  %c1 = arith.constant 1 : index
  %c7999 = arith.constant 7999 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c120 = arith.constant 120 : index
  %c8000 = arith.constant 8000 : index
  %0 = fir.alloca i32 {bindc_name = "new", uniq_name = "_QFtest1Enew"}
  %1:2 = hlfir.declare %0 {uniq_name = "_QFtest1Enew"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.alloca i32 {bindc_name = "old", uniq_name = "_QFtest1Eold"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtest1Eold"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.shape %c8000, %c120, %c3 : (index, index, index) -> !fir.shape<3>
  %5:2 = hlfir.declare %arg0(%4) {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.array<8000x120x3xf32>>, !fir.shape<3>) -> (!fir.ref<!fir.array<8000x120x3xf32>>, !fir.ref<!fir.array<8000x120x3xf32>>)
  %6 = fir.load %3#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> i64
  %8 = fir.shape %c7998, %c120 : (index, index) -> !fir.shape<2>
  %9 = hlfir.designate %5#0 (%c2:%c7999:%c1, %c1:%c120:%c1, %7)  shape %8 : (!fir.ref<!fir.array<8000x120x3xf32>>, index, index, index, index, index, index, i64, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  %10 = hlfir.elemental %8 unordered : (!fir.shape<2>) -> !hlfir.expr<7998x120xf32> {
  ^bb0(%arg1: index, %arg2: index):
    %14 = hlfir.designate %9 (%arg1, %arg2)  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
    %15 = fir.load %14 : !fir.ref<f32>
    %16 = hlfir.no_reassoc %15 : f32
    hlfir.yield_element %16 : f32
  }
  %11 = fir.load %1#0 : !fir.ref<i32>
  %12 = fir.convert %11 : (i32) -> i64
  %13 = hlfir.designate %5#0 (%c2:%c7999:%c1, %c1:%c120:%c1, %12)  shape %8 : (!fir.ref<!fir.array<8000x120x3xf32>>, index, index, index, index, index, index, i64, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  hlfir.assign %10 to %13 : !hlfir.expr<7998x120xf32>, !fir.box<!fir.array<7998x120xf32>>
  hlfir.destroy %10 : !hlfir.expr<7998x120xf32>
  return
}
// CHECK-LABEL:   func.func @_QPtest1(
// CHECK:           fir.do_loop %[[VAL_21:.*]] =
// CHECK:             fir.do_loop %[[VAL_22:.*]] =
// CHECK:               %[[VAL_23:.*]] = hlfir.designate %[[VAL_17:.*]] (%[[VAL_22]], %[[VAL_21]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<f32>
// CHECK:               %[[VAL_25:.*]] = hlfir.no_reassoc %[[VAL_24]] : f32
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_20:.*]] (%[[VAL_22]], %[[VAL_21]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_25]] to %[[VAL_26]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }

func.func @_QPtest2(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>> {fir.bindc_name = "x"}) {
  %c120 = arith.constant 120 : index
  %c7998 = arith.constant 7998 : index
  %c1 = arith.constant 1 : index
  %c7999 = arith.constant 7999 : index
  %c2 = arith.constant 2 : index
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest2Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>)
  %1 = fir.load %0#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>>
  %2 = fir.shape %c7998, %c120 : (index, index) -> !fir.shape<2>
  %3 = hlfir.designate %1 (%c2:%c7999:%c1, %c1:%c120:%c1, %c2)  shape %2 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  %4 = hlfir.elemental %2 unordered : (!fir.shape<2>) -> !hlfir.expr<7998x120xf32> {
  ^bb0(%arg1: index, %arg2: index):
    %6 = hlfir.designate %3 (%arg1, %arg2)  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
    %7 = fir.load %6 : !fir.ref<f32>
    %8 = hlfir.no_reassoc %7 : f32
    hlfir.yield_element %8 : f32
  }
  %5 = hlfir.designate %1 (%c2:%c7999:%c1, %c1:%c120:%c1, %c1)  shape %2 : (!fir.box<!fir.heap<!fir.array<?x?x?xf32>>>, index, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  hlfir.assign %4 to %5 : !hlfir.expr<7998x120xf32>, !fir.box<!fir.array<7998x120xf32>>
  hlfir.destroy %4 : !hlfir.expr<7998x120xf32>
  return
}
// CHECK-LABEL:   func.func @_QPtest2(
// CHECK:           fir.do_loop %[[VAL_11:.*]] =
// CHECK:             fir.do_loop %[[VAL_12:.*]] =
// CHECK:               %[[VAL_13:.*]] = hlfir.designate %[[VAL_9:.*]] (%[[VAL_12]], %[[VAL_11]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<f32>
// CHECK:               %[[VAL_15:.*]] = hlfir.no_reassoc %[[VAL_14]] : f32
// CHECK:               %[[VAL_16:.*]] = hlfir.designate %[[VAL_10:.*]] (%[[VAL_12]], %[[VAL_11]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_15]] to %[[VAL_16]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }

func.func @_QPtest3(%arg0: !fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>> {fir.bindc_name = "x"}) {
  %c7998 = arith.constant 7998 : index
  %c7999 = arith.constant 7999 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c120 = arith.constant 120 : index
  %c8000 = arith.constant 8000 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = fir.alloca i32 {bindc_name = "new", uniq_name = "_QFtest3Enew"}
  %1:2 = hlfir.declare %0 {uniq_name = "_QFtest3Enew"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.alloca i32 {bindc_name = "old", uniq_name = "_QFtest3Eold"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtest3Eold"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg0(%4) {uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>>, !fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>>)
  %6 = hlfir.designate %5#0 (%c1)  : (!fir.ref<!fir.array<10x!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>>, index) -> !fir.ref<!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>
  %7 = fir.shape %c8000, %c120, %c3 : (index, index, index) -> !fir.shape<3>
  %8 = fir.load %3#0 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> i64
  %10 = fir.shape %c7998, %c120 : (index, index) -> !fir.shape<2>
  %11 = hlfir.designate %6{"x"} <%7> (%c2:%c7999:%c1, %c1:%c120:%c1, %9)  shape %10 : (!fir.ref<!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>, !fir.shape<3>, index, index, index, index, index, index, i64, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  %12 = hlfir.elemental %10 unordered : (!fir.shape<2>) -> !hlfir.expr<7998x120xf32> {
  ^bb0(%arg1: index, %arg2: index):
    %16 = hlfir.designate %11 (%arg1, %arg2)  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
    %17 = fir.load %16 : !fir.ref<f32>
    %18 = hlfir.no_reassoc %17 : f32
    hlfir.yield_element %18 : f32
  }
  %13 = fir.load %1#0 : !fir.ref<i32>
  %14 = fir.convert %13 : (i32) -> i64
  %15 = hlfir.designate %6{"x"} <%7> (%c2:%c7999:%c1, %c1:%c120:%c1, %14)  shape %10 : (!fir.ref<!fir.type<_QMtypesTt{x:!fir.array<8000x120x3xf32>}>>, !fir.shape<3>, index, index, index, index, index, index, i64, !fir.shape<2>) -> !fir.box<!fir.array<7998x120xf32>>
  hlfir.assign %12 to %15 : !hlfir.expr<7998x120xf32>, !fir.box<!fir.array<7998x120xf32>>
  hlfir.destroy %12 : !hlfir.expr<7998x120xf32>
  return
}
// CHECK-LABEL:   func.func @_QPtest3(
// CHECK:           fir.do_loop %[[VAL_24:.*]] =
// CHECK:             fir.do_loop %[[VAL_25:.*]] =
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_20:.*]] (%[[VAL_25]], %[[VAL_24]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_27:.*]] = fir.load %[[VAL_26]] : !fir.ref<f32>
// CHECK:               %[[VAL_28:.*]] = hlfir.no_reassoc %[[VAL_27]] : f32
// CHECK:               %[[VAL_29:.*]] = hlfir.designate %[[VAL_23:.*]] (%[[VAL_25]], %[[VAL_24]])  : (!fir.box<!fir.array<7998x120xf32>>, index, index) -> !fir.ref<f32>
// CHECK:               hlfir.assign %[[VAL_28]] to %[[VAL_29]] : f32, !fir.ref<f32>
// CHECK:             }
// CHECK:           }

// ! ub == lb - 1
// subroutine test4(x, i1, i2, nx)
//   real :: x(i2), f
//   do i=i1,i2,nx
//      x(i:i+nx-1) = (x(i-nx:i-1))
//   end do
// end subroutine test4
func.func @_QPtest4(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "i1"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i2"}, %arg3: !fir.ref<i32> {fir.bindc_name = "nx"}) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32 {bindc_name = "f", uniq_name = "_QFtest4Ef"}
  %1:2 = hlfir.declare %0 {uniq_name = "_QFtest4Ef"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest4Ei"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtest4Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg1 {uniq_name = "_QFtest4Ei1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg2 {uniq_name = "_QFtest4Ei2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6:2 = hlfir.declare %arg3 {uniq_name = "_QFtest4Enx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.load %5#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = arith.cmpi sgt, %8, %c0 : index
  %10 = arith.select %9, %8, %c0 : index
  %11 = fir.shape %10 : (index) -> !fir.shape<1>
  %12:2 = hlfir.declare %arg0(%11) {uniq_name = "_QFtest4Ex"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  %13 = fir.load %4#0 : !fir.ref<i32>
  %14 = fir.convert %13 : (i32) -> index
  %15 = fir.load %5#0 : !fir.ref<i32>
  %16 = fir.convert %15 : (i32) -> index
  %17 = fir.load %6#0 : !fir.ref<i32>
  %18 = fir.convert %17 : (i32) -> index
  %19 = fir.convert %14 : (index) -> i32
  %20:2 = fir.do_loop %arg4 = %14 to %16 step %18 iter_args(%arg5 = %19) -> (index, i32) {
    fir.store %arg5 to %3#1 : !fir.ref<i32>
    %21 = fir.load %3#0 : !fir.ref<i32>
    %22 = fir.load %6#0 : !fir.ref<i32>
    %23 = arith.subi %21, %22 : i32
    %24 = arith.subi %21, %c1_i32 : i32
    %25 = fir.convert %23 : (i32) -> index
    %26 = fir.convert %24 : (i32) -> index
    %27 = arith.subi %26, %25 : index
    %28 = arith.addi %27, %c1 : index
    %29 = arith.cmpi sgt, %28, %c0 : index
    %30 = arith.select %29, %28, %c0 : index
    %31 = fir.shape %30 : (index) -> !fir.shape<1>
    %32 = hlfir.designate %12#0 (%25:%26:%c1)  shape %31 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
    %33 = hlfir.elemental %31 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
    ^bb0(%arg6: index):
      %48 = hlfir.designate %32 (%arg6)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      %49 = fir.load %48 : !fir.ref<f32>
      %50 = hlfir.no_reassoc %49 : f32
      hlfir.yield_element %50 : f32
    }
    %34 = arith.addi %21, %22 : i32
    %35 = arith.subi %34, %c1_i32 : i32
    %36 = fir.convert %21 : (i32) -> index
    %37 = fir.convert %35 : (i32) -> index
    %38 = arith.subi %37, %36 : index
    %39 = arith.addi %38, %c1 : index
    %40 = arith.cmpi sgt, %39, %c0 : index
    %41 = arith.select %40, %39, %c0 : index
    %42 = fir.shape %41 : (index) -> !fir.shape<1>
    %43 = hlfir.designate %12#0 (%36:%37:%c1)  shape %42 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
    hlfir.assign %33 to %43 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
    hlfir.destroy %33 : !hlfir.expr<?xf32>
    %44 = arith.addi %arg4, %18 : index
    %45 = fir.convert %18 : (index) -> i32
    %46 = fir.load %3#1 : !fir.ref<i32>
    %47 = arith.addi %46, %45 : i32
    fir.result %44, %47 : index, i32
  }
  fir.store %20#1 to %3#1 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QPtest4(
// CHECK-NOT: hlfir.elemental

// ! lb == ub + 1
// subroutine test5(x, i1, i2, nx)
//   real :: x(i2), f
//   do i=i1,i2,nx
//      x(i+1:i+nx-1) = (x(i-nx:i))
//   end do
// end subroutine test5
func.func @_QPtest5(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "i1"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i2"}, %arg3: !fir.ref<i32> {fir.bindc_name = "nx"}) {
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32 {bindc_name = "f", uniq_name = "_QFtest5Ef"}
  %1:2 = hlfir.declare %0 {uniq_name = "_QFtest5Ef"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest5Ei"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtest5Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg1 {uniq_name = "_QFtest5Ei1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg2 {uniq_name = "_QFtest5Ei2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6:2 = hlfir.declare %arg3 {uniq_name = "_QFtest5Enx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.load %5#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = arith.cmpi sgt, %8, %c0 : index
  %10 = arith.select %9, %8, %c0 : index
  %11 = fir.shape %10 : (index) -> !fir.shape<1>
  %12:2 = hlfir.declare %arg0(%11) {uniq_name = "_QFtest5Ex"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  %13 = fir.load %4#0 : !fir.ref<i32>
  %14 = fir.convert %13 : (i32) -> index
  %15 = fir.load %5#0 : !fir.ref<i32>
  %16 = fir.convert %15 : (i32) -> index
  %17 = fir.load %6#0 : !fir.ref<i32>
  %18 = fir.convert %17 : (i32) -> index
  %19 = fir.convert %14 : (index) -> i32
  %20:2 = fir.do_loop %arg4 = %14 to %16 step %18 iter_args(%arg5 = %19) -> (index, i32) {
    fir.store %arg5 to %3#1 : !fir.ref<i32>
    %21 = fir.load %3#0 : !fir.ref<i32>
    %22 = fir.load %6#0 : !fir.ref<i32>
    %23 = arith.subi %21, %22 : i32
    %24 = fir.convert %23 : (i32) -> index
    %25 = fir.convert %21 : (i32) -> index
    %26 = arith.subi %25, %24 : index
    %27 = arith.addi %26, %c1 : index
    %28 = arith.cmpi sgt, %27, %c0 : index
    %29 = arith.select %28, %27, %c0 : index
    %30 = fir.shape %29 : (index) -> !fir.shape<1>
    %31 = hlfir.designate %12#0 (%24:%25:%c1)  shape %30 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
    %32 = hlfir.elemental %30 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
    ^bb0(%arg6: index):
      %48 = hlfir.designate %31 (%arg6)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      %49 = fir.load %48 : !fir.ref<f32>
      %50 = hlfir.no_reassoc %49 : f32
      hlfir.yield_element %50 : f32
    }
    %33 = arith.addi %21, %c1_i32 : i32
    %34 = arith.addi %21, %22 : i32
    %35 = arith.subi %34, %c1_i32 : i32
    %36 = fir.convert %33 : (i32) -> index
    %37 = fir.convert %35 : (i32) -> index
    %38 = arith.subi %37, %36 : index
    %39 = arith.addi %38, %c1 : index
    %40 = arith.cmpi sgt, %39, %c0 : index
    %41 = arith.select %40, %39, %c0 : index
    %42 = fir.shape %41 : (index) -> !fir.shape<1>
    %43 = hlfir.designate %12#0 (%36:%37:%c1)  shape %42 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
    hlfir.assign %32 to %43 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
    hlfir.destroy %32 : !hlfir.expr<?xf32>
    %44 = arith.addi %arg4, %18 : index
    %45 = fir.convert %18 : (index) -> i32
    %46 = fir.load %3#1 : !fir.ref<i32>
    %47 = arith.addi %46, %45 : i32
    fir.result %44, %47 : index, i32
  }
  fir.store %20#1 to %3#1 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QPtest5(
// CHECK-NOT: hlfir.elemental

// ! ub = lb - 1 and dim1 is unknown
// ! FIR lowering produces a temp.
// subroutine test6(x, i1, i2, nx)
//   real :: x(i2,i2), f
//   integer n1, n2, n3, n4
//   do i=i1,i2,nx
//      x(i:i+nx-1,n1:n2) = (x(i-nx:i-1,n3:n4))
//   end do
// end subroutine test6
func.func @_QPtest6(%arg0: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "i1"}, %arg2: !fir.ref<i32> {fir.bindc_name = "i2"}, %arg3: !fir.ref<i32> {fir.bindc_name = "nx"}) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32 {bindc_name = "f", uniq_name = "_QFtest6Ef"}
  %1:2 = hlfir.declare %0 {uniq_name = "_QFtest6Ef"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest6Ei"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtest6Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:2 = hlfir.declare %arg1 {uniq_name = "_QFtest6Ei1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg2 {uniq_name = "_QFtest6Ei2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.alloca i32 {bindc_name = "n1", uniq_name = "_QFtest6En1"}
  %7:2 = hlfir.declare %6 {uniq_name = "_QFtest6En1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %8 = fir.alloca i32 {bindc_name = "n2", uniq_name = "_QFtest6En2"}
  %9:2 = hlfir.declare %8 {uniq_name = "_QFtest6En2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %10 = fir.alloca i32 {bindc_name = "n3", uniq_name = "_QFtest6En3"}
  %11:2 = hlfir.declare %10 {uniq_name = "_QFtest6En3"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %12 = fir.alloca i32 {bindc_name = "n4", uniq_name = "_QFtest6En4"}
  %13:2 = hlfir.declare %12 {uniq_name = "_QFtest6En4"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %14:2 = hlfir.declare %arg3 {uniq_name = "_QFtest6Enx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %15 = fir.load %5#0 : !fir.ref<i32>
  %16 = fir.convert %15 : (i32) -> index
  %17 = arith.cmpi sgt, %16, %c0 : index
  %18 = arith.select %17, %16, %c0 : index
  %19 = fir.shape %18, %18 : (index, index) -> !fir.shape<2>
  %20:2 = hlfir.declare %arg0(%19) {uniq_name = "_QFtest6Ex"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
  %21 = fir.load %4#0 : !fir.ref<i32>
  %22 = fir.convert %21 : (i32) -> index
  %23 = fir.load %5#0 : !fir.ref<i32>
  %24 = fir.convert %23 : (i32) -> index
  %25 = fir.load %14#0 : !fir.ref<i32>
  %26 = fir.convert %25 : (i32) -> index
  %27 = fir.convert %22 : (index) -> i32
  %28:2 = fir.do_loop %arg4 = %22 to %24 step %26 iter_args(%arg5 = %27) -> (index, i32) {
    fir.store %arg5 to %3#1 : !fir.ref<i32>
    %29 = fir.load %3#0 : !fir.ref<i32>
    %30 = fir.load %14#0 : !fir.ref<i32>
    %31 = arith.subi %29, %30 : i32
    %32 = arith.subi %29, %c1_i32 : i32
    %33 = fir.convert %31 : (i32) -> index
    %34 = fir.convert %32 : (i32) -> index
    %35 = arith.subi %34, %33 : index
    %36 = arith.addi %35, %c1 : index
    %37 = arith.cmpi sgt, %36, %c0 : index
    %38 = arith.select %37, %36, %c0 : index
    %39 = fir.load %11#0 : !fir.ref<i32>
    %40 = fir.load %13#0 : !fir.ref<i32>
    %41 = fir.convert %39 : (i32) -> index
    %42 = fir.convert %40 : (i32) -> index
    %43 = arith.subi %42, %41 : index
    %44 = arith.addi %43, %c1 : index
    %45 = arith.cmpi sgt, %44, %c0 : index
    %46 = arith.select %45, %44, %c0 : index
    %47 = fir.shape %38, %46 : (index, index) -> !fir.shape<2>
    %48 = hlfir.designate %20#0 (%33:%34:%c1, %41:%42:%c1)  shape %47 : (!fir.box<!fir.array<?x?xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
    %49 = hlfir.elemental %47 unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
    ^bb0(%arg6: index, %arg7: index):
      %72 = hlfir.designate %48 (%arg6, %arg7)  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
      %73 = fir.load %72 : !fir.ref<f32>
      %74 = hlfir.no_reassoc %73 : f32
      hlfir.yield_element %74 : f32
    }
    %50 = arith.addi %29, %30 : i32
    %51 = arith.subi %50, %c1_i32 : i32
    %52 = fir.convert %29 : (i32) -> index
    %53 = fir.convert %51 : (i32) -> index
    %54 = arith.subi %53, %52 : index
    %55 = arith.addi %54, %c1 : index
    %56 = arith.cmpi sgt, %55, %c0 : index
    %57 = arith.select %56, %55, %c0 : index
    %58 = fir.load %7#0 : !fir.ref<i32>
    %59 = fir.load %9#0 : !fir.ref<i32>
    %60 = fir.convert %58 : (i32) -> index
    %61 = fir.convert %59 : (i32) -> index
    %62 = arith.subi %61, %60 : index
    %63 = arith.addi %62, %c1 : index
    %64 = arith.cmpi sgt, %63, %c0 : index
    %65 = arith.select %64, %63, %c0 : index
    %66 = fir.shape %57, %65 : (index, index) -> !fir.shape<2>
    %67 = hlfir.designate %20#0 (%52:%53:%c1, %60:%61:%c1)  shape %66 : (!fir.box<!fir.array<?x?xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
    hlfir.assign %49 to %67 : !hlfir.expr<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
    hlfir.destroy %49 : !hlfir.expr<?x?xf32>
    %68 = arith.addi %arg4, %26 : index
    %69 = fir.convert %26 : (index) -> i32
    %70 = fir.load %3#1 : !fir.ref<i32>
    %71 = arith.addi %70, %69 : i32
    fir.result %68, %71 : index, i32
  }
  fir.store %28#1 to %3#1 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QPtest6(
// CHECK-NOT: hlfir.elemental
