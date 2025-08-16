// Test hlfir.eoshift simplification to hlfir.elemental and hlfir.eval_in_mem:
// RUN: fir-opt --simplify-hlfir-intrinsics %s | FileCheck %s

// module eoshift_types
//   type t
//   end type t
// end module eoshift_types
//
// ! Test contiguous 1D array with statically absent boundary.
// subroutine eoshift1(n, array)
//   integer :: n
//   real(2) :: array(n)
//   array = EOSHIFT(array, 2)
// end subroutine
func.func @_QPeoshift1(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?xf16>> {fir.bindc_name = "array"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift1En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.load %1#0 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.cmpi sgt, %3, %c0 : index
  %5 = arith.select %4, %3, %c0 : index
  %6 = fir.shape %5 : (index) -> !fir.shape<1>
  %7:2 = hlfir.declare %arg1(%6) dummy_scope %0 {uniq_name = "_QFeoshift1Earray"} : (!fir.ref<!fir.array<?xf16>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf16>>, !fir.ref<!fir.array<?xf16>>)
  %8 = hlfir.eoshift %7#0 %c2_i32 : (!fir.box<!fir.array<?xf16>>, i32) -> !hlfir.expr<?xf16>
  hlfir.assign %8 to %7#0 : !hlfir.expr<?xf16>, !fir.box<!fir.array<?xf16>>
  hlfir.destroy %8 : !hlfir.expr<?xf16>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift1(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.array<?xf16>> {fir.bindc_name = "array"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_5]] {uniq_name = "_QFeoshift1En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_4]] : index
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_8]], %[[VAL_4]] : index
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[ARG1]](%[[VAL_11]]) dummy_scope %[[VAL_5]] {uniq_name = "_QFeoshift1Earray"} : (!fir.ref<!fir.array<?xf16>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf16>>, !fir.ref<!fir.array<?xf16>>)
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_3]] : (i32) -> i64
// CHECK:           %[[VAL_15:.*]] = hlfir.eval_in_mem shape %[[VAL_11]] : (!fir.shape<1>) -> !hlfir.expr<?xf16> {
// CHECK:           ^bb0(%[[VAL_16:.*]]: !fir.ref<!fir.array<?xf16>>):
// CHECK:             %[[VAL_17:.*]] = fir.embox %[[VAL_16]](%[[VAL_11]]) : (!fir.ref<!fir.array<?xf16>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf16>>
// CHECK:             %[[VAL_18:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_19:.*]] = arith.subi %[[VAL_1]], %[[VAL_14]] overflow<nsw> : i64
// CHECK:             %[[VAL_20:.*]] = arith.select %[[VAL_18]], %[[VAL_19]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_21:.*]] = arith.select %[[VAL_18]], %[[VAL_1]], %[[VAL_14]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.subi %[[VAL_1]], %[[VAL_13]] overflow<nsw> : i64
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] overflow<nsw> : i64
// CHECK:             %[[VAL_24:.*]] = arith.cmpi sgt, %[[VAL_22]], %[[VAL_14]] : i64
// CHECK:             %[[VAL_25:.*]] = arith.select %[[VAL_24]], %[[VAL_1]], %[[VAL_23]] : i64
// CHECK:             %[[VAL_26:.*]] = arith.subi %[[VAL_13]], %[[VAL_14]] overflow<nsw> : i64
// CHECK:             %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_13]], %[[VAL_14]] : i64
// CHECK:             %[[VAL_28:.*]] = arith.select %[[VAL_27]], %[[VAL_1]], %[[VAL_26]] : i64
// CHECK:             %[[VAL_29:.*]] = arith.select %[[VAL_18]], %[[VAL_25]], %[[VAL_28]] : i64
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i64) -> index
// CHECK:             fir.do_loop %[[VAL_31:.*]] = %[[VAL_0]] to %[[VAL_30]] step %[[VAL_0]] unordered {
// CHECK:               %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (index) -> i64
// CHECK:               %[[VAL_33:.*]] = arith.addi %[[VAL_32]], %[[VAL_21]] overflow<nsw> : i64
// CHECK:               %[[VAL_34:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_33]])  : (!fir.box<!fir.array<?xf16>>, i64) -> !fir.ref<f16>
// CHECK:               %[[VAL_35:.*]] = fir.load %[[VAL_34]] : !fir.ref<f16>
// CHECK:               %[[VAL_36:.*]] = arith.addi %[[VAL_32]], %[[VAL_20]] overflow<nsw> : i64
// CHECK:               %[[VAL_37:.*]] = hlfir.designate %[[VAL_17]] (%[[VAL_36]])  : (!fir.box<!fir.array<?xf16>>, i64) -> !fir.ref<f16>
// CHECK:               hlfir.assign %[[VAL_35]] to %[[VAL_37]] : f16, !fir.ref<f16>
// CHECK:             }
// CHECK:             %[[VAL_38:.*]] = arith.subi %[[VAL_13]], %[[VAL_29]] overflow<nsw> : i64
// CHECK:             %[[VAL_39:.*]] = arith.select %[[VAL_18]], %[[VAL_1]], %[[VAL_29]] : i64
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_38]] : (i64) -> index
// CHECK:             fir.do_loop %[[VAL_41:.*]] = %[[VAL_0]] to %[[VAL_40]] step %[[VAL_0]] unordered {
// CHECK:               %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (index) -> i64
// CHECK:               %[[VAL_43:.*]] = arith.addi %[[VAL_42]], %[[VAL_39]] overflow<nsw> : i64
// CHECK:               %[[VAL_44:.*]] = hlfir.designate %[[VAL_17]] (%[[VAL_43]])  : (!fir.box<!fir.array<?xf16>>, i64) -> !fir.ref<f16>
// CHECK:               hlfir.assign %[[VAL_2]] to %[[VAL_44]] : f16, !fir.ref<f16>
// CHECK:             }
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_15]] to %[[VAL_12]]#0 : !hlfir.expr<?xf16>, !fir.box<!fir.array<?xf16>>
// CHECK:           hlfir.destroy %[[VAL_15]] : !hlfir.expr<?xf16>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the scalar constant boundary.
// subroutine eoshift2(n, array)
//   integer :: n
//   logical(2) :: array(n)
//   array = EOSHIFT(array, 2, boundary=.true._2, dim=1)
// end subroutine
func.func @_QPeoshift2(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?x!fir.logical<2>>> {fir.bindc_name = "array"}) {
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift2En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.load %1#0 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.cmpi sgt, %3, %c0 : index
  %5 = arith.select %4, %3, %c0 : index
  %6 = fir.shape %5 : (index) -> !fir.shape<1>
  %7:2 = hlfir.declare %arg1(%6) dummy_scope %0 {uniq_name = "_QFeoshift2Earray"} : (!fir.ref<!fir.array<?x!fir.logical<2>>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<2>>>, !fir.ref<!fir.array<?x!fir.logical<2>>>)
  %8 = fir.convert %true : (i1) -> !fir.logical<2>
  %9 = hlfir.eoshift %7#0 %c2_i32 boundary %8 dim %c1_i32 : (!fir.box<!fir.array<?x!fir.logical<2>>>, i32, !fir.logical<2>, i32) -> !hlfir.expr<?x!fir.logical<2>>
  hlfir.assign %9 to %7#0 : !hlfir.expr<?x!fir.logical<2>>, !fir.box<!fir.array<?x!fir.logical<2>>>
  hlfir.destroy %9 : !hlfir.expr<?x!fir.logical<2>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift2(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.array<?x!fir.logical<2>>> {fir.bindc_name = "array"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_5]] {uniq_name = "_QFeoshift2En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_4]] : index
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_8]], %[[VAL_4]] : index
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[ARG1]](%[[VAL_11]]) dummy_scope %[[VAL_5]] {uniq_name = "_QFeoshift2Earray"} : (!fir.ref<!fir.array<?x!fir.logical<2>>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<2>>>, !fir.ref<!fir.array<?x!fir.logical<2>>>)
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_2]] : (i1) -> !fir.logical<2>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_3]] : (i32) -> i64
// CHECK:           %[[VAL_16:.*]] = hlfir.eval_in_mem shape %[[VAL_11]] : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<2>> {
// CHECK:           ^bb0(%[[VAL_17:.*]]: !fir.ref<!fir.array<?x!fir.logical<2>>>):
// CHECK:             %[[VAL_18:.*]] = fir.embox %[[VAL_17]](%[[VAL_11]]) : (!fir.ref<!fir.array<?x!fir.logical<2>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.logical<2>>>
// CHECK:             %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_20:.*]] = arith.subi %[[VAL_1]], %[[VAL_15]] overflow<nsw> : i64
// CHECK:             %[[VAL_21:.*]] = arith.select %[[VAL_19]], %[[VAL_20]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.select %[[VAL_19]], %[[VAL_1]], %[[VAL_15]] : i64
// CHECK:             %[[VAL_23:.*]] = arith.subi %[[VAL_1]], %[[VAL_14]] overflow<nsw> : i64
// CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_14]], %[[VAL_15]] overflow<nsw> : i64
// CHECK:             %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_23]], %[[VAL_15]] : i64
// CHECK:             %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_1]], %[[VAL_24]] : i64
// CHECK:             %[[VAL_27:.*]] = arith.subi %[[VAL_14]], %[[VAL_15]] overflow<nsw> : i64
// CHECK:             %[[VAL_28:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_15]] : i64
// CHECK:             %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_1]], %[[VAL_27]] : i64
// CHECK:             %[[VAL_30:.*]] = arith.select %[[VAL_19]], %[[VAL_26]], %[[VAL_29]] : i64
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
// CHECK:             fir.do_loop %[[VAL_32:.*]] = %[[VAL_0]] to %[[VAL_31]] step %[[VAL_0]] unordered {
// CHECK:               %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
// CHECK:               %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_22]] overflow<nsw> : i64
// CHECK:               %[[VAL_35:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_34]])  : (!fir.box<!fir.array<?x!fir.logical<2>>>, i64) -> !fir.ref<!fir.logical<2>>
// CHECK:               %[[VAL_36:.*]] = fir.load %[[VAL_35]] : !fir.ref<!fir.logical<2>>
// CHECK:               %[[VAL_37:.*]] = arith.addi %[[VAL_33]], %[[VAL_21]] overflow<nsw> : i64
// CHECK:               %[[VAL_38:.*]] = hlfir.designate %[[VAL_18]] (%[[VAL_37]])  : (!fir.box<!fir.array<?x!fir.logical<2>>>, i64) -> !fir.ref<!fir.logical<2>>
// CHECK:               hlfir.assign %[[VAL_36]] to %[[VAL_38]] : !fir.logical<2>, !fir.ref<!fir.logical<2>>
// CHECK:             }
// CHECK:             %[[VAL_39:.*]] = arith.subi %[[VAL_14]], %[[VAL_30]] overflow<nsw> : i64
// CHECK:             %[[VAL_40:.*]] = arith.select %[[VAL_19]], %[[VAL_1]], %[[VAL_30]] : i64
// CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_39]] : (i64) -> index
// CHECK:             fir.do_loop %[[VAL_42:.*]] = %[[VAL_0]] to %[[VAL_41]] step %[[VAL_0]] unordered {
// CHECK:               %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (index) -> i64
// CHECK:               %[[VAL_44:.*]] = arith.addi %[[VAL_43]], %[[VAL_40]] overflow<nsw> : i64
// CHECK:               %[[VAL_45:.*]] = hlfir.designate %[[VAL_18]] (%[[VAL_44]])  : (!fir.box<!fir.array<?x!fir.logical<2>>>, i64) -> !fir.ref<!fir.logical<2>>
// CHECK:               hlfir.assign %[[VAL_13]] to %[[VAL_45]] : !fir.logical<2>, !fir.ref<!fir.logical<2>>
// CHECK:             }
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_16]] to %[[VAL_12]]#0 : !hlfir.expr<?x!fir.logical<2>>, !fir.box<!fir.array<?x!fir.logical<2>>>
// CHECK:           hlfir.destroy %[[VAL_16]] : !hlfir.expr<?x!fir.logical<2>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the scalar always present boundary.
// subroutine eoshift3(n, array, boundary)
//   integer :: n
//   complex(2) :: array(n), boundary
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift3(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?xcomplex<f16>>> {fir.bindc_name = "array"}, %arg2: !fir.ref<complex<f16>> {fir.bindc_name = "boundary"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift3En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFeoshift3Eboundary"} : (!fir.ref<complex<f16>>, !fir.dscope) -> (!fir.ref<complex<f16>>, !fir.ref<complex<f16>>)
  %3 = fir.load %1#0 : !fir.ref<i32>
  %4 = fir.convert %3 : (i32) -> index
  %5 = arith.cmpi sgt, %4, %c0 : index
  %6 = arith.select %5, %4, %c0 : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8:2 = hlfir.declare %arg1(%7) dummy_scope %0 {uniq_name = "_QFeoshift3Earray"} : (!fir.ref<!fir.array<?xcomplex<f16>>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xcomplex<f16>>>, !fir.ref<!fir.array<?xcomplex<f16>>>)
  %9 = hlfir.eoshift %8#0 %c2_i32 boundary %2#0 : (!fir.box<!fir.array<?xcomplex<f16>>>, i32, !fir.ref<complex<f16>>) -> !hlfir.expr<?xcomplex<f16>>
  hlfir.assign %9 to %8#0 : !hlfir.expr<?xcomplex<f16>>, !fir.box<!fir.array<?xcomplex<f16>>>
  hlfir.destroy %9 : !hlfir.expr<?xcomplex<f16>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift3(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.array<?xcomplex<f16>>> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<complex<f16>> {fir.bindc_name = "boundary"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift3En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift3Eboundary"} : (!fir.ref<complex<f16>>, !fir.dscope) -> (!fir.ref<complex<f16>>, !fir.ref<complex<f16>>)
// CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_3]] : index
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_8]], %[[VAL_3]] : index
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[ARG1]](%[[VAL_11]]) dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift3Earray"} : (!fir.ref<!fir.array<?xcomplex<f16>>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xcomplex<f16>>>, !fir.ref<!fir.array<?xcomplex<f16>>>)
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
// CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<complex<f16>>
// CHECK:           %[[VAL_16:.*]] = hlfir.eval_in_mem shape %[[VAL_11]] : (!fir.shape<1>) -> !hlfir.expr<?xcomplex<f16>> {
// CHECK:           ^bb0(%[[VAL_17:.*]]: !fir.ref<!fir.array<?xcomplex<f16>>>):
// CHECK:             %[[VAL_18:.*]] = fir.embox %[[VAL_17]](%[[VAL_11]]) : (!fir.ref<!fir.array<?xcomplex<f16>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xcomplex<f16>>>
// CHECK:             %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_20:.*]] = arith.subi %[[VAL_1]], %[[VAL_14]] overflow<nsw> : i64
// CHECK:             %[[VAL_21:.*]] = arith.select %[[VAL_19]], %[[VAL_20]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.select %[[VAL_19]], %[[VAL_1]], %[[VAL_14]] : i64
// CHECK:             %[[VAL_23:.*]] = arith.subi %[[VAL_1]], %[[VAL_13]] overflow<nsw> : i64
// CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] overflow<nsw> : i64
// CHECK:             %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_23]], %[[VAL_14]] : i64
// CHECK:             %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_1]], %[[VAL_24]] : i64
// CHECK:             %[[VAL_27:.*]] = arith.subi %[[VAL_13]], %[[VAL_14]] overflow<nsw> : i64
// CHECK:             %[[VAL_28:.*]] = arith.cmpi slt, %[[VAL_13]], %[[VAL_14]] : i64
// CHECK:             %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_1]], %[[VAL_27]] : i64
// CHECK:             %[[VAL_30:.*]] = arith.select %[[VAL_19]], %[[VAL_26]], %[[VAL_29]] : i64
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
// CHECK:             fir.do_loop %[[VAL_32:.*]] = %[[VAL_0]] to %[[VAL_31]] step %[[VAL_0]] unordered {
// CHECK:               %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
// CHECK:               %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_22]] overflow<nsw> : i64
// CHECK:               %[[VAL_35:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_34]])  : (!fir.box<!fir.array<?xcomplex<f16>>>, i64) -> !fir.ref<complex<f16>>
// CHECK:               %[[VAL_36:.*]] = fir.load %[[VAL_35]] : !fir.ref<complex<f16>>
// CHECK:               %[[VAL_37:.*]] = arith.addi %[[VAL_33]], %[[VAL_21]] overflow<nsw> : i64
// CHECK:               %[[VAL_38:.*]] = hlfir.designate %[[VAL_18]] (%[[VAL_37]])  : (!fir.box<!fir.array<?xcomplex<f16>>>, i64) -> !fir.ref<complex<f16>>
// CHECK:               hlfir.assign %[[VAL_36]] to %[[VAL_38]] : complex<f16>, !fir.ref<complex<f16>>
// CHECK:             }
// CHECK:             %[[VAL_39:.*]] = arith.subi %[[VAL_13]], %[[VAL_30]] overflow<nsw> : i64
// CHECK:             %[[VAL_40:.*]] = arith.select %[[VAL_19]], %[[VAL_1]], %[[VAL_30]] : i64
// CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_39]] : (i64) -> index
// CHECK:             fir.do_loop %[[VAL_42:.*]] = %[[VAL_0]] to %[[VAL_41]] step %[[VAL_0]] unordered {
// CHECK:               %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (index) -> i64
// CHECK:               %[[VAL_44:.*]] = arith.addi %[[VAL_43]], %[[VAL_40]] overflow<nsw> : i64
// CHECK:               %[[VAL_45:.*]] = hlfir.designate %[[VAL_18]] (%[[VAL_44]])  : (!fir.box<!fir.array<?xcomplex<f16>>>, i64) -> !fir.ref<complex<f16>>
// CHECK:               hlfir.assign %[[VAL_15]] to %[[VAL_45]] : complex<f16>, !fir.ref<complex<f16>>
// CHECK:             }
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_16]] to %[[VAL_12]]#0 : !hlfir.expr<?xcomplex<f16>>, !fir.box<!fir.array<?xcomplex<f16>>>
// CHECK:           hlfir.destroy %[[VAL_16]] : !hlfir.expr<?xcomplex<f16>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the scalar optional boundary.
// subroutine eoshift4(n, array, boundary)
//   integer :: n
//   logical :: array(n)
//   logical, optional :: boundary
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift4(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "array"}, %arg2: !fir.ref<!fir.logical<4>> {fir.bindc_name = "boundary", fir.optional}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift4En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 dummy_scope %0 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift4Eboundary"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %3 = fir.load %1#0 : !fir.ref<i32>
  %4 = fir.convert %3 : (i32) -> index
  %5 = arith.cmpi sgt, %4, %c0 : index
  %6 = arith.select %5, %4, %c0 : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8:2 = hlfir.declare %arg1(%7) dummy_scope %0 {uniq_name = "_QFeoshift4Earray"} : (!fir.ref<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.ref<!fir.array<?x!fir.logical<4>>>)
  %9 = fir.is_present %2#0 : (!fir.ref<!fir.logical<4>>) -> i1
  %10 = fir.embox %2#0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %11 = fir.absent !fir.box<!fir.logical<4>>
  %12 = arith.select %9, %10, %11 : !fir.box<!fir.logical<4>>
  %13 = hlfir.eoshift %8#0 %c2_i32 boundary %12 : (!fir.box<!fir.array<?x!fir.logical<4>>>, i32, !fir.box<!fir.logical<4>>) -> !hlfir.expr<?x!fir.logical<4>>
  hlfir.assign %13 to %8#0 : !hlfir.expr<?x!fir.logical<4>>, !fir.box<!fir.array<?x!fir.logical<4>>>
  hlfir.destroy %13 : !hlfir.expr<?x!fir.logical<4>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift4(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "boundary", fir.optional}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_5]] {uniq_name = "_QFeoshift4En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_5]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift4Eboundary"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
// CHECK:           %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_4]] : index
// CHECK:           %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_9]], %[[VAL_4]] : index
// CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[ARG1]](%[[VAL_12]]) dummy_scope %[[VAL_5]] {uniq_name = "_QFeoshift4Earray"} : (!fir.ref<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.ref<!fir.array<?x!fir.logical<4>>>)
// CHECK:           %[[VAL_14:.*]] = fir.is_present %[[VAL_7]]#0 : (!fir.ref<!fir.logical<4>>) -> i1
// CHECK:           %[[VAL_15:.*]] = fir.embox %[[VAL_7]]#0 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK:           %[[VAL_16:.*]] = fir.absent !fir.box<!fir.logical<4>>
// CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_14]], %[[VAL_15]], %[[VAL_16]] : !fir.box<!fir.logical<4>>
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_11]] : (index) -> i64
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_3]] : (i32) -> i64
// CHECK:           %[[VAL_20:.*]] = fir.is_present %[[VAL_17]] : (!fir.box<!fir.logical<4>>) -> i1
// CHECK:           %[[VAL_21:.*]] = fir.if %[[VAL_20]] -> (!fir.logical<4>) {
// CHECK:             %[[VAL_22:.*]] = fir.box_addr %[[VAL_17]] : (!fir.box<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>>
// CHECK:             %[[VAL_23:.*]] = fir.load %[[VAL_22]] : !fir.ref<!fir.logical<4>>
// CHECK:             fir.result %[[VAL_23]] : !fir.logical<4>
// CHECK:           } else {
// CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_2]] : (i1) -> !fir.logical<4>
// CHECK:             fir.result %[[VAL_24]] : !fir.logical<4>
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = hlfir.eval_in_mem shape %[[VAL_12]] : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
// CHECK:           ^bb0(%[[VAL_26:.*]]: !fir.ref<!fir.array<?x!fir.logical<4>>>):
// CHECK:             %[[VAL_27:.*]] = fir.embox %[[VAL_26]](%[[VAL_12]]) : (!fir.ref<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.logical<4>>>
// CHECK:             %[[VAL_28:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_29:.*]] = arith.subi %[[VAL_1]], %[[VAL_19]] overflow<nsw> : i64
// CHECK:             %[[VAL_30:.*]] = arith.select %[[VAL_28]], %[[VAL_29]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_31:.*]] = arith.select %[[VAL_28]], %[[VAL_1]], %[[VAL_19]] : i64
// CHECK:             %[[VAL_32:.*]] = arith.subi %[[VAL_1]], %[[VAL_18]] overflow<nsw> : i64
// CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_18]], %[[VAL_19]] overflow<nsw> : i64
// CHECK:             %[[VAL_34:.*]] = arith.cmpi sgt, %[[VAL_32]], %[[VAL_19]] : i64
// CHECK:             %[[VAL_35:.*]] = arith.select %[[VAL_34]], %[[VAL_1]], %[[VAL_33]] : i64
// CHECK:             %[[VAL_36:.*]] = arith.subi %[[VAL_18]], %[[VAL_19]] overflow<nsw> : i64
// CHECK:             %[[VAL_37:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_19]] : i64
// CHECK:             %[[VAL_38:.*]] = arith.select %[[VAL_37]], %[[VAL_1]], %[[VAL_36]] : i64
// CHECK:             %[[VAL_39:.*]] = arith.select %[[VAL_28]], %[[VAL_35]], %[[VAL_38]] : i64
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i64) -> index
// CHECK:             fir.do_loop %[[VAL_41:.*]] = %[[VAL_0]] to %[[VAL_40]] step %[[VAL_0]] unordered {
// CHECK:               %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (index) -> i64
// CHECK:               %[[VAL_43:.*]] = arith.addi %[[VAL_42]], %[[VAL_31]] overflow<nsw> : i64
// CHECK:               %[[VAL_44:.*]] = hlfir.designate %[[VAL_13]]#0 (%[[VAL_43]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_45:.*]] = fir.load %[[VAL_44]] : !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_46:.*]] = arith.addi %[[VAL_42]], %[[VAL_30]] overflow<nsw> : i64
// CHECK:               %[[VAL_47:.*]] = hlfir.designate %[[VAL_27]] (%[[VAL_46]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
// CHECK:               hlfir.assign %[[VAL_45]] to %[[VAL_47]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:             }
// CHECK:             %[[VAL_48:.*]] = arith.subi %[[VAL_18]], %[[VAL_39]] overflow<nsw> : i64
// CHECK:             %[[VAL_49:.*]] = arith.select %[[VAL_28]], %[[VAL_1]], %[[VAL_39]] : i64
// CHECK:             %[[VAL_50:.*]] = fir.convert %[[VAL_48]] : (i64) -> index
// CHECK:             fir.do_loop %[[VAL_51:.*]] = %[[VAL_0]] to %[[VAL_50]] step %[[VAL_0]] unordered {
// CHECK:               %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (index) -> i64
// CHECK:               %[[VAL_53:.*]] = arith.addi %[[VAL_52]], %[[VAL_49]] overflow<nsw> : i64
// CHECK:               %[[VAL_54:.*]] = hlfir.designate %[[VAL_27]] (%[[VAL_53]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
// CHECK:               hlfir.assign %[[VAL_21]] to %[[VAL_54]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:             }
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_25]] to %[[VAL_13]]#0 : !hlfir.expr<?x!fir.logical<4>>, !fir.box<!fir.array<?x!fir.logical<4>>>
// CHECK:           hlfir.destroy %[[VAL_25]] : !hlfir.expr<?x!fir.logical<4>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the array always present boundary.
// subroutine eoshift5(n, array, boundary)
//   integer :: n
//   real :: array(n,n)
//   real :: boundary(:)
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift5(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "array"}, %arg2: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "boundary"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift5En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFeoshift5Eboundary"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %3 = fir.load %1#0 : !fir.ref<i32>
  %4 = fir.convert %3 : (i32) -> index
  %5 = arith.cmpi sgt, %4, %c0 : index
  %6 = arith.select %5, %4, %c0 : index
  %7 = fir.load %1#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = arith.cmpi sgt, %8, %c0 : index
  %10 = arith.select %9, %8, %c0 : index
  %11 = fir.shape %6, %10 : (index, index) -> !fir.shape<2>
  %12:2 = hlfir.declare %arg1(%11) dummy_scope %0 {uniq_name = "_QFeoshift5Earray"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
  %13 = hlfir.eoshift %12#0 %c2_i32 boundary %2#0 : (!fir.box<!fir.array<?x?xf32>>, i32, !fir.box<!fir.array<?xf32>>) -> !hlfir.expr<?x?xf32>
  hlfir.assign %13 to %12#0 : !hlfir.expr<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
  hlfir.destroy %13 : !hlfir.expr<?x?xf32>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift5(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "boundary"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift5En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift5Eboundary"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_3]] : index
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_8]], %[[VAL_3]] : index
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_3]] : index
// CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_12]], %[[VAL_3]] : index
// CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_10]], %[[VAL_14]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[ARG1]](%[[VAL_15]]) dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift5Earray"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
// CHECK:           %[[VAL_19:.*]] = hlfir.eval_in_mem shape %[[VAL_15]] : (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
// CHECK:           ^bb0(%[[VAL_20:.*]]: !fir.ref<!fir.array<?x?xf32>>):
// CHECK:             %[[VAL_21:.*]] = fir.embox %[[VAL_20]](%[[VAL_15]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:             fir.do_loop %[[VAL_22:.*]] = %[[VAL_1]] to %[[VAL_14]] step %[[VAL_1]] unordered {
// CHECK:               %[[VAL_23:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_22]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:               %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<f32>
// CHECK:               %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_0]] : i64
// CHECK:               %[[VAL_26:.*]] = arith.subi %[[VAL_0]], %[[VAL_18]] overflow<nsw> : i64
// CHECK:               %[[VAL_27:.*]] = arith.select %[[VAL_25]], %[[VAL_26]], %[[VAL_0]] : i64
// CHECK:               %[[VAL_28:.*]] = arith.select %[[VAL_25]], %[[VAL_0]], %[[VAL_18]] : i64
// CHECK:               %[[VAL_29:.*]] = arith.subi %[[VAL_0]], %[[VAL_17]] overflow<nsw> : i64
// CHECK:               %[[VAL_30:.*]] = arith.addi %[[VAL_17]], %[[VAL_18]] overflow<nsw> : i64
// CHECK:               %[[VAL_31:.*]] = arith.cmpi sgt, %[[VAL_29]], %[[VAL_18]] : i64
// CHECK:               %[[VAL_32:.*]] = arith.select %[[VAL_31]], %[[VAL_0]], %[[VAL_30]] : i64
// CHECK:               %[[VAL_33:.*]] = arith.subi %[[VAL_17]], %[[VAL_18]] overflow<nsw> : i64
// CHECK:               %[[VAL_34:.*]] = arith.cmpi slt, %[[VAL_17]], %[[VAL_18]] : i64
// CHECK:               %[[VAL_35:.*]] = arith.select %[[VAL_34]], %[[VAL_0]], %[[VAL_33]] : i64
// CHECK:               %[[VAL_36:.*]] = arith.select %[[VAL_25]], %[[VAL_32]], %[[VAL_35]] : i64
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i64) -> index
// CHECK:               fir.do_loop %[[VAL_38:.*]] = %[[VAL_1]] to %[[VAL_37]] step %[[VAL_1]] unordered {
// CHECK:                 %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (index) -> i64
// CHECK:                 %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_28]] overflow<nsw> : i64
// CHECK:                 %[[VAL_41:.*]] = hlfir.designate %[[VAL_16]]#0 (%[[VAL_40]], %[[VAL_22]])  : (!fir.box<!fir.array<?x?xf32>>, i64, index) -> !fir.ref<f32>
// CHECK:                 %[[VAL_42:.*]] = fir.load %[[VAL_41]] : !fir.ref<f32>
// CHECK:                 %[[VAL_43:.*]] = arith.addi %[[VAL_39]], %[[VAL_27]] overflow<nsw> : i64
// CHECK:                 %[[VAL_44:.*]] = hlfir.designate %[[VAL_21]] (%[[VAL_43]], %[[VAL_22]])  : (!fir.box<!fir.array<?x?xf32>>, i64, index) -> !fir.ref<f32>
// CHECK:                 hlfir.assign %[[VAL_42]] to %[[VAL_44]] : f32, !fir.ref<f32>
// CHECK:               }
// CHECK:               %[[VAL_45:.*]] = arith.subi %[[VAL_17]], %[[VAL_36]] overflow<nsw> : i64
// CHECK:               %[[VAL_46:.*]] = arith.select %[[VAL_25]], %[[VAL_0]], %[[VAL_36]] : i64
// CHECK:               %[[VAL_47:.*]] = fir.convert %[[VAL_45]] : (i64) -> index
// CHECK:               fir.do_loop %[[VAL_48:.*]] = %[[VAL_1]] to %[[VAL_47]] step %[[VAL_1]] unordered {
// CHECK:                 %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (index) -> i64
// CHECK:                 %[[VAL_50:.*]] = arith.addi %[[VAL_49]], %[[VAL_46]] overflow<nsw> : i64
// CHECK:                 %[[VAL_51:.*]] = hlfir.designate %[[VAL_21]] (%[[VAL_50]], %[[VAL_22]])  : (!fir.box<!fir.array<?x?xf32>>, i64, index) -> !fir.ref<f32>
// CHECK:                 hlfir.assign %[[VAL_24]] to %[[VAL_51]] : f32, !fir.ref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_19]] to %[[VAL_16]]#0 : !hlfir.expr<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
// CHECK:           hlfir.destroy %[[VAL_19]] : !hlfir.expr<?x?xf32>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the array optional boundary.
// subroutine eoshift6(n, array, boundary)
//   integer :: n
//   real :: array(n,n)
//   real, optional :: boundary(n)
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift6(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "array"}, %arg2: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "boundary", fir.optional}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift6En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.load %1#0 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.cmpi sgt, %3, %c0 : index
  %5 = arith.select %4, %3, %c0 : index
  %6 = fir.load %1#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> index
  %8 = arith.cmpi sgt, %7, %c0 : index
  %9 = arith.select %8, %7, %c0 : index
  %10 = fir.shape %5, %9 : (index, index) -> !fir.shape<2>
  %11:2 = hlfir.declare %arg1(%10) dummy_scope %0 {uniq_name = "_QFeoshift6Earray"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
  %12 = fir.load %1#0 : !fir.ref<i32>
  %13 = fir.convert %12 : (i32) -> index
  %14 = arith.cmpi sgt, %13, %c0 : index
  %15 = arith.select %14, %13, %c0 : index
  %16 = fir.shape %15 : (index) -> !fir.shape<1>
  %17:2 = hlfir.declare %arg2(%16) dummy_scope %0 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift6Eboundary"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  %18 = fir.is_present %17#0 : (!fir.box<!fir.array<?xf32>>) -> i1
  %19 = fir.shape %15 : (index) -> !fir.shape<1>
  %20 = fir.embox %17#1(%19) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  %21 = fir.absent !fir.box<!fir.array<?xf32>>
  %22 = arith.select %18, %20, %21 : !fir.box<!fir.array<?xf32>>
  %23 = hlfir.eoshift %11#0 %c2_i32 boundary %22 : (!fir.box<!fir.array<?x?xf32>>, i32, !fir.box<!fir.array<?xf32>>) -> !hlfir.expr<?x?xf32>
  hlfir.assign %23 to %11#0 : !hlfir.expr<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
  hlfir.destroy %23 : !hlfir.expr<?x?xf32>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift6(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "boundary", fir.optional}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_7]] {uniq_name = "_QFeoshift6En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_6]] : index
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_6]] : index
// CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
// CHECK:           %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_6]] : index
// CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_14]], %[[VAL_6]] : index
// CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_12]], %[[VAL_16]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_18:.*]]:2 = hlfir.declare %[[ARG1]](%[[VAL_17]]) dummy_scope %[[VAL_7]] {uniq_name = "_QFeoshift6Earray"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
// CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> index
// CHECK:           %[[VAL_21:.*]] = arith.cmpi sgt, %[[VAL_20]], %[[VAL_6]] : index
// CHECK:           %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_20]], %[[VAL_6]] : index
// CHECK:           %[[VAL_23:.*]] = fir.shape %[[VAL_22]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_24:.*]]:2 = hlfir.declare %[[ARG2]](%[[VAL_23]]) dummy_scope %[[VAL_7]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift6Eboundary"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
// CHECK:           %[[VAL_25:.*]] = fir.is_present %[[VAL_24]]#0 : (!fir.box<!fir.array<?xf32>>) -> i1
// CHECK:           %[[VAL_26:.*]] = fir.shape %[[VAL_22]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_27:.*]] = fir.embox %[[VAL_24]]#1(%[[VAL_26]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
// CHECK:           %[[VAL_28:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
// CHECK:           %[[VAL_29:.*]] = arith.select %[[VAL_25]], %[[VAL_27]], %[[VAL_28]] : !fir.box<!fir.array<?xf32>>
// CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_12]] : (index) -> i64
// CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
// CHECK:           %[[VAL_32:.*]] = fir.is_present %[[VAL_29]] : (!fir.box<!fir.array<?xf32>>) -> i1
// CHECK:           %[[VAL_33:.*]] = arith.select %[[VAL_32]], %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           %[[VAL_34:.*]] = hlfir.eval_in_mem shape %[[VAL_17]] : (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
// CHECK:           ^bb0(%[[VAL_35:.*]]: !fir.ref<!fir.array<?x?xf32>>):
// CHECK:             %[[VAL_36:.*]] = fir.embox %[[VAL_35]](%[[VAL_17]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:             fir.do_loop %[[VAL_37:.*]] = %[[VAL_1]] to %[[VAL_16]] step %[[VAL_1]] unordered {
// CHECK:               %[[VAL_38:.*]] = fir.if %[[VAL_33]] -> (f32) {
// CHECK:                 fir.result %[[VAL_4]] : f32
// CHECK:               } else {
// CHECK:                 %[[VAL_39:.*]]:3 = fir.box_dims %[[VAL_29]], %[[VAL_6]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:                 %[[VAL_40:.*]] = arith.subi %[[VAL_39]]#0, %[[VAL_1]] overflow<nsw> : index
// CHECK:                 %[[VAL_41:.*]] = arith.addi %[[VAL_37]], %[[VAL_40]] overflow<nsw> : index
// CHECK:                 %[[VAL_42:.*]] = hlfir.designate %[[VAL_29]] (%[[VAL_41]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:                 %[[VAL_43:.*]] = fir.load %[[VAL_42]] : !fir.ref<f32>
// CHECK:                 fir.result %[[VAL_43]] : f32
// CHECK:               }
// CHECK:               %[[VAL_44:.*]] = arith.cmpi slt, %[[VAL_31]], %[[VAL_0]] : i64
// CHECK:               %[[VAL_45:.*]] = arith.subi %[[VAL_0]], %[[VAL_31]] overflow<nsw> : i64
// CHECK:               %[[VAL_46:.*]] = arith.select %[[VAL_44]], %[[VAL_45]], %[[VAL_0]] : i64
// CHECK:               %[[VAL_47:.*]] = arith.select %[[VAL_44]], %[[VAL_0]], %[[VAL_31]] : i64
// CHECK:               %[[VAL_48:.*]] = arith.subi %[[VAL_0]], %[[VAL_30]] overflow<nsw> : i64
// CHECK:               %[[VAL_49:.*]] = arith.addi %[[VAL_30]], %[[VAL_31]] overflow<nsw> : i64
// CHECK:               %[[VAL_50:.*]] = arith.cmpi sgt, %[[VAL_48]], %[[VAL_31]] : i64
// CHECK:               %[[VAL_51:.*]] = arith.select %[[VAL_50]], %[[VAL_0]], %[[VAL_49]] : i64
// CHECK:               %[[VAL_52:.*]] = arith.subi %[[VAL_30]], %[[VAL_31]] overflow<nsw> : i64
// CHECK:               %[[VAL_53:.*]] = arith.cmpi slt, %[[VAL_30]], %[[VAL_31]] : i64
// CHECK:               %[[VAL_54:.*]] = arith.select %[[VAL_53]], %[[VAL_0]], %[[VAL_52]] : i64
// CHECK:               %[[VAL_55:.*]] = arith.select %[[VAL_44]], %[[VAL_51]], %[[VAL_54]] : i64
// CHECK:               %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (i64) -> index
// CHECK:               fir.do_loop %[[VAL_57:.*]] = %[[VAL_1]] to %[[VAL_56]] step %[[VAL_1]] unordered {
// CHECK:                 %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (index) -> i64
// CHECK:                 %[[VAL_59:.*]] = arith.addi %[[VAL_58]], %[[VAL_47]] overflow<nsw> : i64
// CHECK:                 %[[VAL_60:.*]] = hlfir.designate %[[VAL_18]]#0 (%[[VAL_59]], %[[VAL_37]])  : (!fir.box<!fir.array<?x?xf32>>, i64, index) -> !fir.ref<f32>
// CHECK:                 %[[VAL_61:.*]] = fir.load %[[VAL_60]] : !fir.ref<f32>
// CHECK:                 %[[VAL_62:.*]] = arith.addi %[[VAL_58]], %[[VAL_46]] overflow<nsw> : i64
// CHECK:                 %[[VAL_63:.*]] = hlfir.designate %[[VAL_36]] (%[[VAL_62]], %[[VAL_37]])  : (!fir.box<!fir.array<?x?xf32>>, i64, index) -> !fir.ref<f32>
// CHECK:                 hlfir.assign %[[VAL_61]] to %[[VAL_63]] : f32, !fir.ref<f32>
// CHECK:               }
// CHECK:               %[[VAL_64:.*]] = arith.subi %[[VAL_30]], %[[VAL_55]] overflow<nsw> : i64
// CHECK:               %[[VAL_65:.*]] = arith.select %[[VAL_44]], %[[VAL_0]], %[[VAL_55]] : i64
// CHECK:               %[[VAL_66:.*]] = fir.convert %[[VAL_64]] : (i64) -> index
// CHECK:               fir.do_loop %[[VAL_67:.*]] = %[[VAL_1]] to %[[VAL_66]] step %[[VAL_1]] unordered {
// CHECK:                 %[[VAL_68:.*]] = fir.convert %[[VAL_67]] : (index) -> i64
// CHECK:                 %[[VAL_69:.*]] = arith.addi %[[VAL_68]], %[[VAL_65]] overflow<nsw> : i64
// CHECK:                 %[[VAL_70:.*]] = hlfir.designate %[[VAL_36]] (%[[VAL_69]], %[[VAL_37]])  : (!fir.box<!fir.array<?x?xf32>>, i64, index) -> !fir.ref<f32>
// CHECK:                 hlfir.assign %[[VAL_38]] to %[[VAL_70]] : f32, !fir.ref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_34]] to %[[VAL_18]]#0 : !hlfir.expr<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
// CHECK:           hlfir.destroy %[[VAL_34]] : !hlfir.expr<?x?xf32>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the array expression boundary.
// subroutine eoshift7(n, array)
//   interface
//      function real_boundary(n)
//        integer :: n
//        real :: real_boundary(n)
//      end function
//   end interface
//   integer :: n
//   real :: array(n,n)
//   array = EOSHIFT(array, 2, real_boundary(n))
// end subroutine
func.func @_QPeoshift7(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "array"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift7En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.load %1#0 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.cmpi sgt, %3, %c0 : index
  %5 = arith.select %4, %3, %c0 : index
  %6 = fir.load %1#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> index
  %8 = arith.cmpi sgt, %7, %c0 : index
  %9 = arith.select %8, %7, %c0 : index
  %10 = fir.shape %5, %9 : (index, index) -> !fir.shape<2>
  %11:2 = hlfir.declare %arg1(%10) dummy_scope %0 {uniq_name = "_QFeoshift7Earray"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
  %12:2 = hlfir.declare %1#0 {uniq_name = "_QFeoshift7Freal_boundaryEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %13 = fir.load %12#0 : !fir.ref<i32>
  %14 = fir.convert %13 : (i32) -> index
  %15 = arith.cmpi sgt, %14, %c0 : index
  %16 = arith.select %15, %14, %c0 : index
  %17 = fir.shape %16 : (index) -> !fir.shape<1>
  %18 = hlfir.eval_in_mem shape %17 : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%arg2: !fir.ref<!fir.array<?xf32>>):
    %20 = fir.call @_QPreal_boundary(%1#0) fastmath<contract> : (!fir.ref<i32>) -> !fir.array<?xf32>
    fir.save_result %20 to %arg2(%17) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
  }
  %19 = hlfir.eoshift %11#0 %c2_i32 boundary %18 : (!fir.box<!fir.array<?x?xf32>>, i32, !hlfir.expr<?xf32>) -> !hlfir.expr<?x?xf32>
  hlfir.assign %19 to %11#0 : !hlfir.expr<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
  hlfir.destroy %19 : !hlfir.expr<?x?xf32>
  hlfir.destroy %18 : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift7(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "array"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift7En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
// CHECK:           %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_3]] : index
// CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_7]], %[[VAL_3]] : index
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
// CHECK:           %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_3]] : index
// CHECK:           %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_11]], %[[VAL_3]] : index
// CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_9]], %[[VAL_13]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[ARG1]](%[[VAL_14]]) dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift7Earray"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
// CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_5]]#0 {uniq_name = "_QFeoshift7Freal_boundaryEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_16]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
// CHECK:           %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_18]], %[[VAL_3]] : index
// CHECK:           %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_18]], %[[VAL_3]] : index
// CHECK:           %[[VAL_21:.*]] = fir.shape %[[VAL_20]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_22:.*]] = hlfir.eval_in_mem shape %[[VAL_21]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:           ^bb0(%[[VAL_23:.*]]: !fir.ref<!fir.array<?xf32>>):
// CHECK:             %[[VAL_24:.*]] = fir.call @_QPreal_boundary(%[[VAL_5]]#0) fastmath<contract> : (!fir.ref<i32>) -> !fir.array<?xf32>
// CHECK:             fir.save_result %[[VAL_24]] to %[[VAL_23]](%[[VAL_21]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
// CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
// CHECK:           %[[VAL_27:.*]] = hlfir.eval_in_mem shape %[[VAL_14]] : (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
// CHECK:           ^bb0(%[[VAL_28:.*]]: !fir.ref<!fir.array<?x?xf32>>):
// CHECK:             %[[VAL_29:.*]] = fir.embox %[[VAL_28]](%[[VAL_14]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
// CHECK:             fir.do_loop %[[VAL_30:.*]] = %[[VAL_1]] to %[[VAL_13]] step %[[VAL_1]] unordered {
// CHECK:               %[[VAL_31:.*]] = hlfir.apply %[[VAL_22]], %[[VAL_30]] : (!hlfir.expr<?xf32>, index) -> f32
// CHECK:               %[[VAL_32:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_0]] : i64
// CHECK:               %[[VAL_33:.*]] = arith.subi %[[VAL_0]], %[[VAL_26]] overflow<nsw> : i64
// CHECK:               %[[VAL_34:.*]] = arith.select %[[VAL_32]], %[[VAL_33]], %[[VAL_0]] : i64
// CHECK:               %[[VAL_35:.*]] = arith.select %[[VAL_32]], %[[VAL_0]], %[[VAL_26]] : i64
// CHECK:               %[[VAL_36:.*]] = arith.subi %[[VAL_0]], %[[VAL_25]] overflow<nsw> : i64
// CHECK:               %[[VAL_37:.*]] = arith.addi %[[VAL_25]], %[[VAL_26]] overflow<nsw> : i64
// CHECK:               %[[VAL_38:.*]] = arith.cmpi sgt, %[[VAL_36]], %[[VAL_26]] : i64
// CHECK:               %[[VAL_39:.*]] = arith.select %[[VAL_38]], %[[VAL_0]], %[[VAL_37]] : i64
// CHECK:               %[[VAL_40:.*]] = arith.subi %[[VAL_25]], %[[VAL_26]] overflow<nsw> : i64
// CHECK:               %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_25]], %[[VAL_26]] : i64
// CHECK:               %[[VAL_42:.*]] = arith.select %[[VAL_41]], %[[VAL_0]], %[[VAL_40]] : i64
// CHECK:               %[[VAL_43:.*]] = arith.select %[[VAL_32]], %[[VAL_39]], %[[VAL_42]] : i64
// CHECK:               %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i64) -> index
// CHECK:               fir.do_loop %[[VAL_45:.*]] = %[[VAL_1]] to %[[VAL_44]] step %[[VAL_1]] unordered {
// CHECK:                 %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (index) -> i64
// CHECK:                 %[[VAL_47:.*]] = arith.addi %[[VAL_46]], %[[VAL_35]] overflow<nsw> : i64
// CHECK:                 %[[VAL_48:.*]] = hlfir.designate %[[VAL_15]]#0 (%[[VAL_47]], %[[VAL_30]])  : (!fir.box<!fir.array<?x?xf32>>, i64, index) -> !fir.ref<f32>
// CHECK:                 %[[VAL_49:.*]] = fir.load %[[VAL_48]] : !fir.ref<f32>
// CHECK:                 %[[VAL_50:.*]] = arith.addi %[[VAL_46]], %[[VAL_34]] overflow<nsw> : i64
// CHECK:                 %[[VAL_51:.*]] = hlfir.designate %[[VAL_29]] (%[[VAL_50]], %[[VAL_30]])  : (!fir.box<!fir.array<?x?xf32>>, i64, index) -> !fir.ref<f32>
// CHECK:                 hlfir.assign %[[VAL_49]] to %[[VAL_51]] : f32, !fir.ref<f32>
// CHECK:               }
// CHECK:               %[[VAL_52:.*]] = arith.subi %[[VAL_25]], %[[VAL_43]] overflow<nsw> : i64
// CHECK:               %[[VAL_53:.*]] = arith.select %[[VAL_32]], %[[VAL_0]], %[[VAL_43]] : i64
// CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_52]] : (i64) -> index
// CHECK:               fir.do_loop %[[VAL_55:.*]] = %[[VAL_1]] to %[[VAL_54]] step %[[VAL_1]] unordered {
// CHECK:                 %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (index) -> i64
// CHECK:                 %[[VAL_57:.*]] = arith.addi %[[VAL_56]], %[[VAL_53]] overflow<nsw> : i64
// CHECK:                 %[[VAL_58:.*]] = hlfir.designate %[[VAL_29]] (%[[VAL_57]], %[[VAL_30]])  : (!fir.box<!fir.array<?x?xf32>>, i64, index) -> !fir.ref<f32>
// CHECK:                 hlfir.assign %[[VAL_31]] to %[[VAL_58]] : f32, !fir.ref<f32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_27]] to %[[VAL_15]]#0 : !hlfir.expr<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
// CHECK:           hlfir.destroy %[[VAL_27]] : !hlfir.expr<?x?xf32>
// CHECK:           hlfir.destroy %[[VAL_22]] : !hlfir.expr<?xf32>
// CHECK:           return
// CHECK:         }

// ! Tests for CHARACTER type (lowered via hlfir.elemental).

// ! Test contiguous 1D array with statically absent boundary.
// ! CHARACTER with constant length.
// subroutine eoshift1c(n, array)
//   integer :: n
//   character(10,1) :: array(n)
//   array = EOSHIFT(array, 2)
// end subroutine
func.func @_QPeoshift1c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "array"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift1cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,10>>>
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.cmpi sgt, %5, %c0 : index
  %7 = arith.select %6, %5, %c0 : index
  %8 = fir.shape %7 : (index) -> !fir.shape<1>
  %9:2 = hlfir.declare %3(%8) typeparams %c10 dummy_scope %0 {uniq_name = "_QFeoshift1cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>)
  %10 = hlfir.eoshift %9#0 %c2_i32 : (!fir.box<!fir.array<?x!fir.char<1,10>>>, i32) -> !hlfir.expr<?x!fir.char<1,10>>
  hlfir.assign %10 to %9#0 : !hlfir.expr<?x!fir.char<1,10>>, !fir.box<!fir.array<?x!fir.char<1,10>>>
  hlfir.destroy %10 : !hlfir.expr<?x!fir.char<1,10>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift1c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "array"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift1cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,10>>>
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
// CHECK:           %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_2]] : index
// CHECK:           %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_9]], %[[VAL_2]] : index
// CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_12]]) typeparams %[[VAL_3]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift1cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>)
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_11]] : (index) -> i64
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_16:.*]] = fir.alloca !fir.char<1,0> {bindc_name = ".chrtmp"}
// CHECK:           %[[VAL_17:.*]] = fir.emboxchar %[[VAL_16]], %[[VAL_2]] : (!fir.ref<!fir.char<1,0>>, index) -> !fir.boxchar<1>
// CHECK:           %[[VAL_18:.*]] = hlfir.elemental %[[VAL_12]] typeparams %[[VAL_3]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,10>> {
// CHECK:           ^bb0(%[[VAL_19:.*]]: index):
// CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
// CHECK:             %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_15]] overflow<nsw> : i64
// CHECK:             %[[VAL_22:.*]] = arith.cmpi sge, %[[VAL_21]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_23:.*]] = arith.cmpi sle, %[[VAL_21]], %[[VAL_14]] : i64
// CHECK:             %[[VAL_24:.*]] = arith.andi %[[VAL_22]], %[[VAL_23]] : i1
// CHECK:             %[[VAL_25:.*]] = fir.if %[[VAL_24]] -> (!fir.boxchar<1>) {
// CHECK:               %[[VAL_26:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
// CHECK:               %[[VAL_27:.*]] = hlfir.designate %[[VAL_13]]#0 (%[[VAL_26]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
// CHECK:               %[[VAL_28:.*]] = fir.emboxchar %[[VAL_27]], %[[VAL_3]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_28]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_17]] : !fir.boxchar<1>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_25]] : !fir.boxchar<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_18]] to %[[VAL_13]]#0 : !hlfir.expr<?x!fir.char<1,10>>, !fir.box<!fir.array<?x!fir.char<1,10>>>
// CHECK:           hlfir.destroy %[[VAL_18]] : !hlfir.expr<?x!fir.char<1,10>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with statically absent boundary.
// ! CHARACTER with variable length.
// subroutine eoshift2c(n, array)
//   integer :: n
//   character(n,1) :: array(n)
//   array = EOSHIFT(array, 2)
// end subroutine
func.func @_QPeoshift2c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "array"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift2cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = arith.cmpi sgt, %4, %c0_i32 : i32
  %6 = arith.select %5, %4, %c0_i32 : i32
  %7 = fir.load %1#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = arith.cmpi sgt, %8, %c0 : index
  %10 = arith.select %9, %8, %c0 : index
  %11 = fir.shape %10 : (index) -> !fir.shape<1>
  %12:2 = hlfir.declare %3(%11) typeparams %6 dummy_scope %0 {uniq_name = "_QFeoshift2cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
  %13 = hlfir.eoshift %12#0 %c2_i32 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i32) -> !hlfir.expr<?x!fir.char<1,?>>
  hlfir.assign %13 to %12#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
  hlfir.destroy %13 : !hlfir.expr<?x!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift2c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "array"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift2cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_2]] : index
// CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_12]], %[[VAL_2]] : index
// CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_15]]) typeparams %[[VAL_10]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift2cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_14]] : (index) -> i64
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_19:.*]] = fir.alloca !fir.char<1,0> {bindc_name = ".chrtmp"}
// CHECK:           %[[VAL_20:.*]] = fir.emboxchar %[[VAL_19]], %[[VAL_2]] : (!fir.ref<!fir.char<1,0>>, index) -> !fir.boxchar<1>
// CHECK:           %[[VAL_21:.*]] = hlfir.elemental %[[VAL_15]] typeparams %[[VAL_10]] unordered : (!fir.shape<1>, i32) -> !hlfir.expr<?x!fir.char<1,?>> {
// CHECK:           ^bb0(%[[VAL_22:.*]]: index):
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (index) -> i64
// CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_18]] overflow<nsw> : i64
// CHECK:             %[[VAL_25:.*]] = arith.cmpi sge, %[[VAL_24]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_26:.*]] = arith.cmpi sle, %[[VAL_24]], %[[VAL_17]] : i64
// CHECK:             %[[VAL_27:.*]] = arith.andi %[[VAL_25]], %[[VAL_26]] : i1
// CHECK:             %[[VAL_28:.*]] = fir.if %[[VAL_27]] -> (!fir.boxchar<1>) {
// CHECK:               %[[VAL_29:.*]] = fir.convert %[[VAL_24]] : (i64) -> index
// CHECK:               %[[VAL_30:.*]] = hlfir.designate %[[VAL_16]]#0 (%[[VAL_29]])  typeparams %[[VAL_10]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, i32) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_30]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_20]] : !fir.boxchar<1>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_28]] : !fir.boxchar<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_21]] to %[[VAL_16]]#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           hlfir.destroy %[[VAL_21]] : !hlfir.expr<?x!fir.char<1,?>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with statically absent boundary.
// ! CHARACTER with assumed length.
// subroutine eoshift3c(n, array)
//   integer :: n
//   character(*,1) :: array(n)
//   array = EOSHIFT(array, 2)
// end subroutine
func.func @_QPeoshift3c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "array"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift3cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.cmpi sgt, %5, %c0 : index
  %7 = arith.select %6, %5, %c0 : index
  %8 = fir.shape %7 : (index) -> !fir.shape<1>
  %9:2 = hlfir.declare %3(%8) typeparams %2#1 dummy_scope %0 {uniq_name = "_QFeoshift3cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
  %10 = hlfir.eoshift %9#0 %c2_i32 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i32) -> !hlfir.expr<?x!fir.char<1,?>>
  hlfir.assign %10 to %9#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
  hlfir.destroy %10 : !hlfir.expr<?x!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift3c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "array"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_3]] {uniq_name = "_QFeoshift3cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_2]] : index
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_8]], %[[VAL_2]] : index
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_11]]) typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_3]] {uniq_name = "_QFeoshift3cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_15:.*]] = fir.alloca !fir.char<1,0> {bindc_name = ".chrtmp"}
// CHECK:           %[[VAL_16:.*]] = fir.emboxchar %[[VAL_15]], %[[VAL_2]] : (!fir.ref<!fir.char<1,0>>, index) -> !fir.boxchar<1>
// CHECK:           %[[VAL_17:.*]] = hlfir.elemental %[[VAL_11]] typeparams %[[VAL_5]]#1 unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
// CHECK:           ^bb0(%[[VAL_18:.*]]: index):
// CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (index) -> i64
// CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_14]] overflow<nsw> : i64
// CHECK:             %[[VAL_21:.*]] = arith.cmpi sge, %[[VAL_20]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.cmpi sle, %[[VAL_20]], %[[VAL_13]] : i64
// CHECK:             %[[VAL_23:.*]] = arith.andi %[[VAL_21]], %[[VAL_22]] : i1
// CHECK:             %[[VAL_24:.*]] = fir.if %[[VAL_23]] -> (!fir.boxchar<1>) {
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_25]])  typeparams %[[VAL_5]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_26]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_16]] : !fir.boxchar<1>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_24]] : !fir.boxchar<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_17]] to %[[VAL_12]]#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           hlfir.destroy %[[VAL_17]] : !hlfir.expr<?x!fir.char<1,?>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with scalar constant boundary.
// subroutine eoshift4c(n, array)
//   integer :: n
//   character(10,1) :: array(n)
//   array = EOSHIFT(array, 2, '0123456789')
// end subroutine
func.func @_QPeoshift4c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "array"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift4cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,10>>>
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.cmpi sgt, %5, %c0 : index
  %7 = arith.select %6, %5, %c0 : index
  %8 = fir.shape %7 : (index) -> !fir.shape<1>
  %9:2 = hlfir.declare %3(%8) typeparams %c10 dummy_scope %0 {uniq_name = "_QFeoshift4cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>)
  %10 = fir.address_of(@_QQclX30313233343536373839) : !fir.ref<!fir.char<1,10>>
  %11:2 = hlfir.declare %10 typeparams %c10 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX30313233343536373839"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
  %12 = hlfir.eoshift %9#0 %c2_i32 boundary %11#0 : (!fir.box<!fir.array<?x!fir.char<1,10>>>, i32, !fir.ref<!fir.char<1,10>>) -> !hlfir.expr<?x!fir.char<1,10>>
  hlfir.assign %12 to %9#0 : !hlfir.expr<?x!fir.char<1,10>>, !fir.box<!fir.array<?x!fir.char<1,10>>>
  hlfir.destroy %12 : !hlfir.expr<?x!fir.char<1,10>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift4c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "array"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift4cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,10>>>
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
// CHECK:           %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_2]] : index
// CHECK:           %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_9]], %[[VAL_2]] : index
// CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_12]]) typeparams %[[VAL_3]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift4cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>)
// CHECK:           %[[VAL_14:.*]] = fir.address_of(@_QQclX30313233343536373839) : !fir.ref<!fir.char<1,10>>
// CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_14]] typeparams %[[VAL_3]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX30313233343536373839"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_11]] : (index) -> i64
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_18:.*]] = fir.emboxchar %[[VAL_15]]#0, %[[VAL_3]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
// CHECK:           %[[VAL_19:.*]] = hlfir.elemental %[[VAL_12]] typeparams %[[VAL_3]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,10>> {
// CHECK:           ^bb0(%[[VAL_20:.*]]: index):
// CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (index) -> i64
// CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_17]] overflow<nsw> : i64
// CHECK:             %[[VAL_23:.*]] = arith.cmpi sge, %[[VAL_22]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_24:.*]] = arith.cmpi sle, %[[VAL_22]], %[[VAL_16]] : i64
// CHECK:             %[[VAL_25:.*]] = arith.andi %[[VAL_23]], %[[VAL_24]] : i1
// CHECK:             %[[VAL_26:.*]] = fir.if %[[VAL_25]] -> (!fir.boxchar<1>) {
// CHECK:               %[[VAL_27:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
// CHECK:               %[[VAL_28:.*]] = hlfir.designate %[[VAL_13]]#0 (%[[VAL_27]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
// CHECK:               %[[VAL_29:.*]] = fir.emboxchar %[[VAL_28]], %[[VAL_3]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_29]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_18]] : !fir.boxchar<1>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_26]] : !fir.boxchar<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_19]] to %[[VAL_13]]#0 : !hlfir.expr<?x!fir.char<1,10>>, !fir.box<!fir.array<?x!fir.char<1,10>>>
// CHECK:           hlfir.destroy %[[VAL_19]] : !hlfir.expr<?x!fir.char<1,10>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with scalar always present boundary.
// ! CHARACTER with constant length.
// subroutine eoshift5c(n, array, boundary)
//   integer :: n
//   character(10,1) :: array(n), boundary
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift5c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "array"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "boundary"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift5cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,10>>
  %4:2 = hlfir.declare %3 typeparams %c10 dummy_scope %0 {uniq_name = "_QFeoshift5cEboundary"} : (!fir.ref<!fir.char<1,10>>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
  %5:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %6 = fir.convert %5#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,10>>>
  %7 = fir.load %1#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = arith.cmpi sgt, %8, %c0 : index
  %10 = arith.select %9, %8, %c0 : index
  %11 = fir.shape %10 : (index) -> !fir.shape<1>
  %12:2 = hlfir.declare %6(%11) typeparams %c10 dummy_scope %0 {uniq_name = "_QFeoshift5cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>)
  %13 = hlfir.eoshift %12#0 %c2_i32 boundary %4#0 : (!fir.box<!fir.array<?x!fir.char<1,10>>>, i32, !fir.ref<!fir.char<1,10>>) -> !hlfir.expr<?x!fir.char<1,10>>
  hlfir.assign %13 to %12#0 : !hlfir.expr<?x!fir.char<1,10>>, !fir.box<!fir.array<?x!fir.char<1,10>>>
  hlfir.destroy %13 : !hlfir.expr<?x!fir.char<1,10>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift5c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "boundary"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift5cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,10>>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_3]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift5cEboundary"} : (!fir.ref<!fir.char<1,10>>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
// CHECK:           %[[VAL_9:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,10>>>
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_2]] : index
// CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_12]], %[[VAL_2]] : index
// CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_15]]) typeparams %[[VAL_3]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift5cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>)
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_14]] : (index) -> i64
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_19:.*]] = fir.emboxchar %[[VAL_8]]#0, %[[VAL_3]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
// CHECK:           %[[VAL_20:.*]] = hlfir.elemental %[[VAL_15]] typeparams %[[VAL_3]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,10>> {
// CHECK:           ^bb0(%[[VAL_21:.*]]: index):
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (index) -> i64
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_18]] overflow<nsw> : i64
// CHECK:             %[[VAL_24:.*]] = arith.cmpi sge, %[[VAL_23]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_25:.*]] = arith.cmpi sle, %[[VAL_23]], %[[VAL_17]] : i64
// CHECK:             %[[VAL_26:.*]] = arith.andi %[[VAL_24]], %[[VAL_25]] : i1
// CHECK:             %[[VAL_27:.*]] = fir.if %[[VAL_26]] -> (!fir.boxchar<1>) {
// CHECK:               %[[VAL_28:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
// CHECK:               %[[VAL_29:.*]] = hlfir.designate %[[VAL_16]]#0 (%[[VAL_28]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
// CHECK:               %[[VAL_30:.*]] = fir.emboxchar %[[VAL_29]], %[[VAL_3]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_30]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_19]] : !fir.boxchar<1>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_27]] : !fir.boxchar<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_20]] to %[[VAL_16]]#0 : !hlfir.expr<?x!fir.char<1,10>>, !fir.box<!fir.array<?x!fir.char<1,10>>>
// CHECK:           hlfir.destroy %[[VAL_20]] : !hlfir.expr<?x!fir.char<1,10>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with scalar always present boundary.
// ! CHARACTER with variable length.
// subroutine eoshift6c(n, array, boundary)
//   integer :: n
//   character(n,1) :: array(n), boundary
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift6c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "array"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "boundary"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift6cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = arith.cmpi sgt, %4, %c0_i32 : i32
  %6 = arith.select %5, %4, %c0_i32 : i32
  %7 = fir.load %1#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = arith.cmpi sgt, %8, %c0 : index
  %10 = arith.select %9, %8, %c0 : index
  %11 = fir.shape %10 : (index) -> !fir.shape<1>
  %12:2 = hlfir.declare %3(%11) typeparams %6 dummy_scope %0 {uniq_name = "_QFeoshift6cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
  %13:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %14 = fir.load %1#0 : !fir.ref<i32>
  %15 = arith.cmpi sgt, %14, %c0_i32 : i32
  %16 = arith.select %15, %14, %c0_i32 : i32
  %17:2 = hlfir.declare %13#0 typeparams %16 dummy_scope %0 {uniq_name = "_QFeoshift6cEboundary"} : (!fir.ref<!fir.char<1,?>>, i32, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %18 = hlfir.eoshift %12#0 %c2_i32 boundary %17#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i32, !fir.boxchar<1>) -> !hlfir.expr<?x!fir.char<1,?>>
  hlfir.assign %18 to %12#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
  hlfir.destroy %18 : !hlfir.expr<?x!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift6c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "boundary"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift6cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_2]] : index
// CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_12]], %[[VAL_2]] : index
// CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_15]]) typeparams %[[VAL_10]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift6cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
// CHECK:           %[[VAL_17:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_18]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_18]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_21:.*]]:2 = hlfir.declare %[[VAL_17]]#0 typeparams %[[VAL_20]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift6cEboundary"} : (!fir.ref<!fir.char<1,?>>, i32, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_14]] : (index) -> i64
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_24:.*]] = hlfir.elemental %[[VAL_15]] typeparams %[[VAL_10]] unordered : (!fir.shape<1>, i32) -> !hlfir.expr<?x!fir.char<1,?>> {
// CHECK:           ^bb0(%[[VAL_25:.*]]: index):
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (index) -> i64
// CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_23]] overflow<nsw> : i64
// CHECK:             %[[VAL_28:.*]] = arith.cmpi sge, %[[VAL_27]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_29:.*]] = arith.cmpi sle, %[[VAL_27]], %[[VAL_22]] : i64
// CHECK:             %[[VAL_30:.*]] = arith.andi %[[VAL_28]], %[[VAL_29]] : i1
// CHECK:             %[[VAL_31:.*]] = fir.if %[[VAL_30]] -> (!fir.boxchar<1>) {
// CHECK:               %[[VAL_32:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
// CHECK:               %[[VAL_33:.*]] = hlfir.designate %[[VAL_16]]#0 (%[[VAL_32]])  typeparams %[[VAL_10]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, i32) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_33]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_21]]#0 : !fir.boxchar<1>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_31]] : !fir.boxchar<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_24]] to %[[VAL_16]]#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           hlfir.destroy %[[VAL_24]] : !hlfir.expr<?x!fir.char<1,?>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with scalar always present boundary.
// ! CHARACTER with assumed length.
// subroutine eoshift7c(n, array, boundary)
//   integer :: n
//   character(*,1) :: array(n), boundary
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift7c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "array"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "boundary"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift7cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 dummy_scope %0 {uniq_name = "_QFeoshift7cEboundary"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5 = fir.convert %4#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  %6 = fir.load %1#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> index
  %8 = arith.cmpi sgt, %7, %c0 : index
  %9 = arith.select %8, %7, %c0 : index
  %10 = fir.shape %9 : (index) -> !fir.shape<1>
  %11:2 = hlfir.declare %5(%10) typeparams %4#1 dummy_scope %0 {uniq_name = "_QFeoshift7cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
  %12 = hlfir.eoshift %11#0 %c2_i32 boundary %3#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, i32, !fir.boxchar<1>) -> !hlfir.expr<?x!fir.char<1,?>>
  hlfir.assign %12 to %11#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
  hlfir.destroy %12 : !hlfir.expr<?x!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift7c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "boundary"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_3]] {uniq_name = "_QFeoshift7cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_3]] {uniq_name = "_QFeoshift7cEboundary"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_7:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_2]] : index
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_2]] : index
// CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_13]]) typeparams %[[VAL_7]]#1 dummy_scope %[[VAL_3]] {uniq_name = "_QFeoshift7cEarray"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_12]] : (index) -> i64
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_17:.*]] = hlfir.elemental %[[VAL_13]] typeparams %[[VAL_7]]#1 unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
// CHECK:           ^bb0(%[[VAL_18:.*]]: index):
// CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (index) -> i64
// CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_16]] overflow<nsw> : i64
// CHECK:             %[[VAL_21:.*]] = arith.cmpi sge, %[[VAL_20]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.cmpi sle, %[[VAL_20]], %[[VAL_15]] : i64
// CHECK:             %[[VAL_23:.*]] = arith.andi %[[VAL_21]], %[[VAL_22]] : i1
// CHECK:             %[[VAL_24:.*]] = fir.if %[[VAL_23]] -> (!fir.boxchar<1>) {
// CHECK:               %[[VAL_25:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
// CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_14]]#0 (%[[VAL_25]])  typeparams %[[VAL_7]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_26]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_6]]#0 : !fir.boxchar<1>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_24]] : !fir.boxchar<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_17]] to %[[VAL_14]]#0 : !hlfir.expr<?x!fir.char<1,?>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           hlfir.destroy %[[VAL_17]] : !hlfir.expr<?x!fir.char<1,?>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the scalar optional boundary.
// ! CHARACTER with constant length.
// subroutine eoshift8c(n, array, boundary)
//   integer :: n
//   character(10,2) :: array(n)
//   character(10,2), optional :: boundary
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift8c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<2> {fir.bindc_name = "array"}, %arg2: !fir.boxchar<2> {fir.bindc_name = "boundary", fir.optional}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift8cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg2 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.char<2,10>>
  %4:2 = hlfir.declare %3 typeparams %c10 dummy_scope %0 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift8cEboundary"} : (!fir.ref<!fir.char<2,10>>, index, !fir.dscope) -> (!fir.ref<!fir.char<2,10>>, !fir.ref<!fir.char<2,10>>)
  %5:2 = fir.unboxchar %arg1 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
  %6 = fir.convert %5#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<?x!fir.char<2,10>>>
  %7 = fir.load %1#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = arith.cmpi sgt, %8, %c0 : index
  %10 = arith.select %9, %8, %c0 : index
  %11 = fir.shape %10 : (index) -> !fir.shape<1>
  %12:2 = hlfir.declare %6(%11) typeparams %c10 dummy_scope %0 {uniq_name = "_QFeoshift8cEarray"} : (!fir.ref<!fir.array<?x!fir.char<2,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<2,10>>>, !fir.ref<!fir.array<?x!fir.char<2,10>>>)
  %13 = fir.is_present %4#0 : (!fir.ref<!fir.char<2,10>>) -> i1
  %14 = fir.embox %4#0 : (!fir.ref<!fir.char<2,10>>) -> !fir.box<!fir.char<2,10>>
  %15 = fir.absent !fir.box<!fir.char<2,10>>
  %16 = arith.select %13, %14, %15 : !fir.box<!fir.char<2,10>>
  %17 = hlfir.eoshift %12#0 %c2_i32 boundary %16 : (!fir.box<!fir.array<?x!fir.char<2,10>>>, i32, !fir.box<!fir.char<2,10>>) -> !hlfir.expr<?x!fir.char<2,10>>
  hlfir.assign %17 to %12#0 : !hlfir.expr<?x!fir.char<2,10>>, !fir.box<!fir.array<?x!fir.char<2,10>>>
  hlfir.destroy %17 : !hlfir.expr<?x!fir.char<2,10>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift8c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<2> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<2> {fir.bindc_name = "boundary", fir.optional}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift8cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.char<2,10>>
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[VAL_3]] dummy_scope %[[VAL_4]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift8cEboundary"} : (!fir.ref<!fir.char<2,10>>, index, !fir.dscope) -> (!fir.ref<!fir.char<2,10>>, !fir.ref<!fir.char<2,10>>)
// CHECK:           %[[VAL_9:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<?x!fir.char<2,10>>>
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_2]] : index
// CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_12]], %[[VAL_2]] : index
// CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_15]]) typeparams %[[VAL_3]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift8cEarray"} : (!fir.ref<!fir.array<?x!fir.char<2,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<2,10>>>, !fir.ref<!fir.array<?x!fir.char<2,10>>>)
// CHECK:           %[[VAL_17:.*]] = fir.is_present %[[VAL_8]]#0 : (!fir.ref<!fir.char<2,10>>) -> i1
// CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_8]]#0 : (!fir.ref<!fir.char<2,10>>) -> !fir.box<!fir.char<2,10>>
// CHECK:           %[[VAL_19:.*]] = fir.absent !fir.box<!fir.char<2,10>>
// CHECK:           %[[VAL_20:.*]] = arith.select %[[VAL_17]], %[[VAL_18]], %[[VAL_19]] : !fir.box<!fir.char<2,10>>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_14]] : (index) -> i64
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_23:.*]] = fir.is_present %[[VAL_20]] : (!fir.box<!fir.char<2,10>>) -> i1
// CHECK:           %[[VAL_24:.*]] = fir.if %[[VAL_23]] -> (!fir.boxchar<2>) {
// CHECK:             %[[VAL_25:.*]] = fir.box_addr %[[VAL_20]] : (!fir.box<!fir.char<2,10>>) -> !fir.ref<!fir.char<2,10>>
// CHECK:             %[[VAL_26:.*]] = fir.emboxchar %[[VAL_25]], %[[VAL_3]] : (!fir.ref<!fir.char<2,10>>, index) -> !fir.boxchar<2>
// CHECK:             fir.result %[[VAL_26]] : !fir.boxchar<2>
// CHECK:           } else {
// CHECK:             %[[VAL_27:.*]] = fir.alloca !fir.char<2,0> {bindc_name = ".chrtmp"}
// CHECK:             %[[VAL_28:.*]] = fir.emboxchar %[[VAL_27]], %[[VAL_2]] : (!fir.ref<!fir.char<2,0>>, index) -> !fir.boxchar<2>
// CHECK:             fir.result %[[VAL_28]] : !fir.boxchar<2>
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = hlfir.elemental %[[VAL_15]] typeparams %[[VAL_3]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<2,10>> {
// CHECK:           ^bb0(%[[VAL_30:.*]]: index):
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
// CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_31]], %[[VAL_22]] overflow<nsw> : i64
// CHECK:             %[[VAL_33:.*]] = arith.cmpi sge, %[[VAL_32]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_34:.*]] = arith.cmpi sle, %[[VAL_32]], %[[VAL_21]] : i64
// CHECK:             %[[VAL_35:.*]] = arith.andi %[[VAL_33]], %[[VAL_34]] : i1
// CHECK:             %[[VAL_36:.*]] = fir.if %[[VAL_35]] -> (!fir.boxchar<2>) {
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
// CHECK:               %[[VAL_38:.*]] = hlfir.designate %[[VAL_16]]#0 (%[[VAL_37]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<2,10>>>, index, index) -> !fir.ref<!fir.char<2,10>>
// CHECK:               %[[VAL_39:.*]] = fir.emboxchar %[[VAL_38]], %[[VAL_3]] : (!fir.ref<!fir.char<2,10>>, index) -> !fir.boxchar<2>
// CHECK:               fir.result %[[VAL_39]] : !fir.boxchar<2>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_24]] : !fir.boxchar<2>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_36]] : !fir.boxchar<2>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_29]] to %[[VAL_16]]#0 : !hlfir.expr<?x!fir.char<2,10>>, !fir.box<!fir.array<?x!fir.char<2,10>>>
// CHECK:           hlfir.destroy %[[VAL_29]] : !hlfir.expr<?x!fir.char<2,10>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the scalar optional boundary.
// ! CHARACTER with variable length.
// subroutine eoshift9c(n, array, boundary)
//   integer :: n
//   character(n,2) :: array(n)
//   character(n,2), optional :: boundary
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift9c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<2> {fir.bindc_name = "array"}, %arg2: !fir.boxchar<2> {fir.bindc_name = "boundary", fir.optional}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift9cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<?x!fir.char<2,?>>>
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = arith.cmpi sgt, %4, %c0_i32 : i32
  %6 = arith.select %5, %4, %c0_i32 : i32
  %7 = fir.load %1#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = arith.cmpi sgt, %8, %c0 : index
  %10 = arith.select %9, %8, %c0 : index
  %11 = fir.shape %10 : (index) -> !fir.shape<1>
  %12:2 = hlfir.declare %3(%11) typeparams %6 dummy_scope %0 {uniq_name = "_QFeoshift9cEarray"} : (!fir.ref<!fir.array<?x!fir.char<2,?>>>, !fir.shape<1>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<2,?>>>, !fir.ref<!fir.array<?x!fir.char<2,?>>>)
  %13:2 = fir.unboxchar %arg2 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
  %14 = fir.load %1#0 : !fir.ref<i32>
  %15 = arith.cmpi sgt, %14, %c0_i32 : i32
  %16 = arith.select %15, %14, %c0_i32 : i32
  %17:2 = hlfir.declare %13#0 typeparams %16 dummy_scope %0 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift9cEboundary"} : (!fir.ref<!fir.char<2,?>>, i32, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
  %18 = fir.is_present %17#0 : (!fir.boxchar<2>) -> i1
  %19 = fir.embox %17#1 typeparams %16 : (!fir.ref<!fir.char<2,?>>, i32) -> !fir.box<!fir.char<2,?>>
  %20 = fir.absent !fir.box<!fir.char<2,?>>
  %21 = arith.select %18, %19, %20 : !fir.box<!fir.char<2,?>>
  %22 = hlfir.eoshift %12#0 %c2_i32 boundary %21 : (!fir.box<!fir.array<?x!fir.char<2,?>>>, i32, !fir.box<!fir.char<2,?>>) -> !hlfir.expr<?x!fir.char<2,?>>
  hlfir.assign %22 to %12#0 : !hlfir.expr<?x!fir.char<2,?>>, !fir.box<!fir.array<?x!fir.char<2,?>>>
  hlfir.destroy %22 : !hlfir.expr<?x!fir.char<2,?>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift9c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<2> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<2> {fir.bindc_name = "boundary", fir.optional}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_5]] {uniq_name = "_QFeoshift9cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_7:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<?x!fir.char<2,?>>>
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_11:.*]] = arith.select %[[VAL_10]], %[[VAL_9]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
// CHECK:           %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_3]] : index
// CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_13]], %[[VAL_3]] : index
// CHECK:           %[[VAL_16:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_16]]) typeparams %[[VAL_11]] dummy_scope %[[VAL_5]] {uniq_name = "_QFeoshift9cEarray"} : (!fir.ref<!fir.array<?x!fir.char<2,?>>>, !fir.shape<1>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<2,?>>>, !fir.ref<!fir.array<?x!fir.char<2,?>>>)
// CHECK:           %[[VAL_18:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_19]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_22:.*]]:2 = hlfir.declare %[[VAL_18]]#0 typeparams %[[VAL_21]] dummy_scope %[[VAL_5]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift9cEboundary"} : (!fir.ref<!fir.char<2,?>>, i32, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
// CHECK:           %[[VAL_23:.*]] = fir.is_present %[[VAL_22]]#0 : (!fir.boxchar<2>) -> i1
// CHECK:           %[[VAL_24:.*]] = fir.embox %[[VAL_22]]#1 typeparams %[[VAL_21]] : (!fir.ref<!fir.char<2,?>>, i32) -> !fir.box<!fir.char<2,?>>
// CHECK:           %[[VAL_25:.*]] = fir.absent !fir.box<!fir.char<2,?>>
// CHECK:           %[[VAL_26:.*]] = arith.select %[[VAL_23]], %[[VAL_24]], %[[VAL_25]] : !fir.box<!fir.char<2,?>>
// CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_15]] : (index) -> i64
// CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
// CHECK:           %[[VAL_29:.*]] = fir.is_present %[[VAL_26]] : (!fir.box<!fir.char<2,?>>) -> i1
// CHECK:           %[[VAL_30:.*]] = fir.if %[[VAL_29]] -> (!fir.boxchar<2>) {
// CHECK:             %[[VAL_31:.*]] = fir.box_addr %[[VAL_26]] : (!fir.box<!fir.char<2,?>>) -> !fir.ref<!fir.char<2,?>>
// CHECK:             %[[VAL_32:.*]] = fir.box_elesize %[[VAL_26]] : (!fir.box<!fir.char<2,?>>) -> index
// CHECK:             %[[VAL_33:.*]] = arith.divsi %[[VAL_32]], %[[VAL_1]] : index
// CHECK:             %[[VAL_34:.*]] = fir.emboxchar %[[VAL_31]], %[[VAL_33]] : (!fir.ref<!fir.char<2,?>>, index) -> !fir.boxchar<2>
// CHECK:             fir.result %[[VAL_34]] : !fir.boxchar<2>
// CHECK:           } else {
// CHECK:             %[[VAL_35:.*]] = fir.alloca !fir.char<2,0> {bindc_name = ".chrtmp"}
// CHECK:             %[[VAL_36:.*]] = fir.emboxchar %[[VAL_35]], %[[VAL_3]] : (!fir.ref<!fir.char<2,0>>, index) -> !fir.boxchar<2>
// CHECK:             fir.result %[[VAL_36]] : !fir.boxchar<2>
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = hlfir.elemental %[[VAL_16]] typeparams %[[VAL_11]] unordered : (!fir.shape<1>, i32) -> !hlfir.expr<?x!fir.char<2,?>> {
// CHECK:           ^bb0(%[[VAL_38:.*]]: index):
// CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (index) -> i64
// CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_28]] overflow<nsw> : i64
// CHECK:             %[[VAL_41:.*]] = arith.cmpi sge, %[[VAL_40]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_42:.*]] = arith.cmpi sle, %[[VAL_40]], %[[VAL_27]] : i64
// CHECK:             %[[VAL_43:.*]] = arith.andi %[[VAL_41]], %[[VAL_42]] : i1
// CHECK:             %[[VAL_44:.*]] = fir.if %[[VAL_43]] -> (!fir.boxchar<2>) {
// CHECK:               %[[VAL_45:.*]] = fir.convert %[[VAL_40]] : (i64) -> index
// CHECK:               %[[VAL_46:.*]] = hlfir.designate %[[VAL_17]]#0 (%[[VAL_45]])  typeparams %[[VAL_11]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>, index, i32) -> !fir.boxchar<2>
// CHECK:               fir.result %[[VAL_46]] : !fir.boxchar<2>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_30]] : !fir.boxchar<2>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_44]] : !fir.boxchar<2>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_37]] to %[[VAL_17]]#0 : !hlfir.expr<?x!fir.char<2,?>>, !fir.box<!fir.array<?x!fir.char<2,?>>>
// CHECK:           hlfir.destroy %[[VAL_37]] : !hlfir.expr<?x!fir.char<2,?>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the scalar optional boundary.
// ! CHARACTER with assumed length.
// subroutine eoshift10c(n, array, boundary)
//   integer :: n
//   character(*,2) :: array(n)
//   character(*,2), optional :: boundary
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift10c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<2> {fir.bindc_name = "array"}, %arg2: !fir.boxchar<2> {fir.bindc_name = "boundary", fir.optional}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift10cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg2 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift10cEboundary"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
  %4:2 = fir.unboxchar %arg1 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
  %5 = fir.convert %4#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<?x!fir.char<2,?>>>
  %6 = fir.load %1#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> index
  %8 = arith.cmpi sgt, %7, %c0 : index
  %9 = arith.select %8, %7, %c0 : index
  %10 = fir.shape %9 : (index) -> !fir.shape<1>
  %11:2 = hlfir.declare %5(%10) typeparams %4#1 dummy_scope %0 {uniq_name = "_QFeoshift10cEarray"} : (!fir.ref<!fir.array<?x!fir.char<2,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<2,?>>>, !fir.ref<!fir.array<?x!fir.char<2,?>>>)
  %12 = fir.is_present %3#0 : (!fir.boxchar<2>) -> i1
  %13 = fir.embox %3#1 typeparams %2#1 : (!fir.ref<!fir.char<2,?>>, index) -> !fir.box<!fir.char<2,?>>
  %14 = fir.absent !fir.box<!fir.char<2,?>>
  %15 = arith.select %12, %13, %14 : !fir.box<!fir.char<2,?>>
  %16 = hlfir.eoshift %11#0 %c2_i32 boundary %15 : (!fir.box<!fir.array<?x!fir.char<2,?>>>, i32, !fir.box<!fir.char<2,?>>) -> !hlfir.expr<?x!fir.char<2,?>>
  hlfir.assign %16 to %11#0 : !hlfir.expr<?x!fir.char<2,?>>, !fir.box<!fir.array<?x!fir.char<2,?>>>
  hlfir.destroy %16 : !hlfir.expr<?x!fir.char<2,?>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift10c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<2> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<2> {fir.bindc_name = "boundary", fir.optional}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift10cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 dummy_scope %[[VAL_4]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift10cEboundary"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
// CHECK:           %[[VAL_8:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<?x!fir.char<2,?>>>
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
// CHECK:           %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_3]] : index
// CHECK:           %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_11]], %[[VAL_3]] : index
// CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_14]]) typeparams %[[VAL_8]]#1 dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift10cEarray"} : (!fir.ref<!fir.array<?x!fir.char<2,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<2,?>>>, !fir.ref<!fir.array<?x!fir.char<2,?>>>)
// CHECK:           %[[VAL_16:.*]] = fir.is_present %[[VAL_7]]#0 : (!fir.boxchar<2>) -> i1
// CHECK:           %[[VAL_17:.*]] = fir.embox %[[VAL_7]]#1 typeparams %[[VAL_6]]#1 : (!fir.ref<!fir.char<2,?>>, index) -> !fir.box<!fir.char<2,?>>
// CHECK:           %[[VAL_18:.*]] = fir.absent !fir.box<!fir.char<2,?>>
// CHECK:           %[[VAL_19:.*]] = arith.select %[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : !fir.box<!fir.char<2,?>>
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_13]] : (index) -> i64
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
// CHECK:           %[[VAL_22:.*]] = fir.is_present %[[VAL_19]] : (!fir.box<!fir.char<2,?>>) -> i1
// CHECK:           %[[VAL_23:.*]] = fir.if %[[VAL_22]] -> (!fir.boxchar<2>) {
// CHECK:             %[[VAL_24:.*]] = fir.box_addr %[[VAL_19]] : (!fir.box<!fir.char<2,?>>) -> !fir.ref<!fir.char<2,?>>
// CHECK:             %[[VAL_25:.*]] = fir.box_elesize %[[VAL_19]] : (!fir.box<!fir.char<2,?>>) -> index
// CHECK:             %[[VAL_26:.*]] = arith.divsi %[[VAL_25]], %[[VAL_1]] : index
// CHECK:             %[[VAL_27:.*]] = fir.emboxchar %[[VAL_24]], %[[VAL_26]] : (!fir.ref<!fir.char<2,?>>, index) -> !fir.boxchar<2>
// CHECK:             fir.result %[[VAL_27]] : !fir.boxchar<2>
// CHECK:           } else {
// CHECK:             %[[VAL_28:.*]] = fir.alloca !fir.char<2,0> {bindc_name = ".chrtmp"}
// CHECK:             %[[VAL_29:.*]] = fir.emboxchar %[[VAL_28]], %[[VAL_3]] : (!fir.ref<!fir.char<2,0>>, index) -> !fir.boxchar<2>
// CHECK:             fir.result %[[VAL_29]] : !fir.boxchar<2>
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = hlfir.elemental %[[VAL_14]] typeparams %[[VAL_8]]#1 unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<2,?>> {
// CHECK:           ^bb0(%[[VAL_31:.*]]: index):
// CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (index) -> i64
// CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_32]], %[[VAL_21]] overflow<nsw> : i64
// CHECK:             %[[VAL_34:.*]] = arith.cmpi sge, %[[VAL_33]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_35:.*]] = arith.cmpi sle, %[[VAL_33]], %[[VAL_20]] : i64
// CHECK:             %[[VAL_36:.*]] = arith.andi %[[VAL_34]], %[[VAL_35]] : i1
// CHECK:             %[[VAL_37:.*]] = fir.if %[[VAL_36]] -> (!fir.boxchar<2>) {
// CHECK:               %[[VAL_38:.*]] = fir.convert %[[VAL_33]] : (i64) -> index
// CHECK:               %[[VAL_39:.*]] = hlfir.designate %[[VAL_15]]#0 (%[[VAL_38]])  typeparams %[[VAL_8]]#1 : (!fir.box<!fir.array<?x!fir.char<2,?>>>, index, index) -> !fir.boxchar<2>
// CHECK:               fir.result %[[VAL_39]] : !fir.boxchar<2>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_23]] : !fir.boxchar<2>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_37]] : !fir.boxchar<2>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_30]] to %[[VAL_15]]#0 : !hlfir.expr<?x!fir.char<2,?>>, !fir.box<!fir.array<?x!fir.char<2,?>>>
// CHECK:           hlfir.destroy %[[VAL_30]] : !hlfir.expr<?x!fir.char<2,?>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the array always present boundary.
// ! CHARACTER with constant length.
// subroutine eoshift11c(n, array, boundary)
//   integer :: n
//   character(10,4) :: array(n,n), boundary(:)
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift11c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<4> {fir.bindc_name = "array"}, %arg2: !fir.box<!fir.array<?x!fir.char<4,10>>> {fir.bindc_name = "boundary"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift11cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 typeparams %c10 dummy_scope %0 {uniq_name = "_QFeoshift11cEboundary"} : (!fir.box<!fir.array<?x!fir.char<4,10>>>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<4,10>>>, !fir.box<!fir.array<?x!fir.char<4,10>>>)
  %3:2 = fir.unboxchar %arg1 : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
  %4 = fir.convert %3#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<4,10>>>
  %5 = fir.load %1#0 : !fir.ref<i32>
  %6 = fir.convert %5 : (i32) -> index
  %7 = arith.cmpi sgt, %6, %c0 : index
  %8 = arith.select %7, %6, %c0 : index
  %9 = fir.load %1#0 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = arith.cmpi sgt, %10, %c0 : index
  %12 = arith.select %11, %10, %c0 : index
  %13 = fir.shape %8, %12 : (index, index) -> !fir.shape<2>
  %14:2 = hlfir.declare %4(%13) typeparams %c10 dummy_scope %0 {uniq_name = "_QFeoshift11cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<4,10>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<4,10>>>, !fir.ref<!fir.array<?x?x!fir.char<4,10>>>)
  %15 = hlfir.eoshift %14#0 %c2_i32 boundary %2#0 : (!fir.box<!fir.array<?x?x!fir.char<4,10>>>, i32, !fir.box<!fir.array<?x!fir.char<4,10>>>) -> !hlfir.expr<?x?x!fir.char<4,10>>
  hlfir.assign %15 to %14#0 : !hlfir.expr<?x?x!fir.char<4,10>>, !fir.box<!fir.array<?x?x!fir.char<4,10>>>
  hlfir.destroy %15 : !hlfir.expr<?x?x!fir.char<4,10>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift11c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<4> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.char<4,10>>> {fir.bindc_name = "boundary"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift11cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[ARG2]] typeparams %[[VAL_3]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift11cEboundary"} : (!fir.box<!fir.array<?x!fir.char<4,10>>>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<4,10>>>, !fir.box<!fir.array<?x!fir.char<4,10>>>)
// CHECK:           %[[VAL_7:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<4,10>>>
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_2]] : index
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_2]] : index
// CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
// CHECK:           %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_2]] : index
// CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_14]], %[[VAL_2]] : index
// CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_12]], %[[VAL_16]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_17]]) typeparams %[[VAL_3]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift11cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<4,10>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<4,10>>>, !fir.ref<!fir.array<?x?x!fir.char<4,10>>>)
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_12]] : (index) -> i64
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_21:.*]] = hlfir.elemental %[[VAL_17]] typeparams %[[VAL_3]] unordered : (!fir.shape<2>, index) -> !hlfir.expr<?x?x!fir.char<4,10>> {
// CHECK:           ^bb0(%[[VAL_22:.*]]: index, %[[VAL_23:.*]]: index):
// CHECK:             %[[VAL_24:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_23]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x!fir.char<4,10>>>, index, index) -> !fir.ref<!fir.char<4,10>>
// CHECK:             %[[VAL_25:.*]] = fir.emboxchar %[[VAL_24]], %[[VAL_3]] : (!fir.ref<!fir.char<4,10>>, index) -> !fir.boxchar<4>
// CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_22]] : (index) -> i64
// CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_20]] overflow<nsw> : i64
// CHECK:             %[[VAL_28:.*]] = arith.cmpi sge, %[[VAL_27]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_29:.*]] = arith.cmpi sle, %[[VAL_27]], %[[VAL_19]] : i64
// CHECK:             %[[VAL_30:.*]] = arith.andi %[[VAL_28]], %[[VAL_29]] : i1
// CHECK:             %[[VAL_31:.*]] = fir.if %[[VAL_30]] -> (!fir.boxchar<4>) {
// CHECK:               %[[VAL_32:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
// CHECK:               %[[VAL_33:.*]] = hlfir.designate %[[VAL_18]]#0 (%[[VAL_32]], %[[VAL_23]])  typeparams %[[VAL_3]] : (!fir.box<!fir.array<?x?x!fir.char<4,10>>>, index, index, index) -> !fir.ref<!fir.char<4,10>>
// CHECK:               %[[VAL_34:.*]] = fir.emboxchar %[[VAL_33]], %[[VAL_3]] : (!fir.ref<!fir.char<4,10>>, index) -> !fir.boxchar<4>
// CHECK:               fir.result %[[VAL_34]] : !fir.boxchar<4>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_25]] : !fir.boxchar<4>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_31]] : !fir.boxchar<4>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_21]] to %[[VAL_18]]#0 : !hlfir.expr<?x?x!fir.char<4,10>>, !fir.box<!fir.array<?x?x!fir.char<4,10>>>
// CHECK:           hlfir.destroy %[[VAL_21]] : !hlfir.expr<?x?x!fir.char<4,10>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the array always present boundary.
// ! CHARACTER with variable length.
// subroutine eoshift12c(n, array, boundary)
//   integer :: n
//   character(n,4) :: array(n,n), boundary(:)
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift12c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<4> {fir.bindc_name = "array"}, %arg2: !fir.box<!fir.array<?x!fir.char<4,?>>> {fir.bindc_name = "boundary"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift12cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<4,?>>>
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = arith.cmpi sgt, %4, %c0_i32 : i32
  %6 = arith.select %5, %4, %c0_i32 : i32
  %7 = fir.load %1#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = arith.cmpi sgt, %8, %c0 : index
  %10 = arith.select %9, %8, %c0 : index
  %11 = fir.load %1#0 : !fir.ref<i32>
  %12 = fir.convert %11 : (i32) -> index
  %13 = arith.cmpi sgt, %12, %c0 : index
  %14 = arith.select %13, %12, %c0 : index
  %15 = fir.shape %10, %14 : (index, index) -> !fir.shape<2>
  %16:2 = hlfir.declare %3(%15) typeparams %6 dummy_scope %0 {uniq_name = "_QFeoshift12cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<4,?>>>, !fir.shape<2>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<4,?>>>, !fir.ref<!fir.array<?x?x!fir.char<4,?>>>)
  %17 = fir.load %1#0 : !fir.ref<i32>
  %18 = arith.cmpi sgt, %17, %c0_i32 : i32
  %19 = arith.select %18, %17, %c0_i32 : i32
  %20:2 = hlfir.declare %arg2 typeparams %19 dummy_scope %0 {uniq_name = "_QFeoshift12cEboundary"} : (!fir.box<!fir.array<?x!fir.char<4,?>>>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<4,?>>>, !fir.box<!fir.array<?x!fir.char<4,?>>>)
  %21 = hlfir.eoshift %16#0 %c2_i32 boundary %20#0 : (!fir.box<!fir.array<?x?x!fir.char<4,?>>>, i32, !fir.box<!fir.array<?x!fir.char<4,?>>>) -> !hlfir.expr<?x?x!fir.char<4,?>>
  hlfir.assign %21 to %16#0 : !hlfir.expr<?x?x!fir.char<4,?>>, !fir.box<!fir.array<?x?x!fir.char<4,?>>>
  hlfir.destroy %21 : !hlfir.expr<?x?x!fir.char<4,?>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift12c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<4> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.char<4,?>>> {fir.bindc_name = "boundary"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift12cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<4,?>>>
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_2]] : index
// CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_12]], %[[VAL_2]] : index
// CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
// CHECK:           %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_2]] : index
// CHECK:           %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_16]], %[[VAL_2]] : index
// CHECK:           %[[VAL_19:.*]] = fir.shape %[[VAL_14]], %[[VAL_18]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_19]]) typeparams %[[VAL_10]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift12cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<4,?>>>, !fir.shape<2>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<4,?>>>, !fir.ref<!fir.array<?x?x!fir.char<4,?>>>)
// CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_22:.*]] = arith.cmpi sgt, %[[VAL_21]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_21]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_24:.*]]:2 = hlfir.declare %[[ARG2]] typeparams %[[VAL_23]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift12cEboundary"} : (!fir.box<!fir.array<?x!fir.char<4,?>>>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<4,?>>>, !fir.box<!fir.array<?x!fir.char<4,?>>>)
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_14]] : (index) -> i64
// CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
// CHECK:           %[[VAL_27:.*]] = hlfir.elemental %[[VAL_19]] typeparams %[[VAL_10]] unordered : (!fir.shape<2>, i32) -> !hlfir.expr<?x?x!fir.char<4,?>> {
// CHECK:           ^bb0(%[[VAL_28:.*]]: index, %[[VAL_29:.*]]: index):
// CHECK:             %[[VAL_30:.*]] = hlfir.designate %[[VAL_24]]#0 (%[[VAL_29]])  typeparams %[[VAL_23]] : (!fir.box<!fir.array<?x!fir.char<4,?>>>, index, i32) -> !fir.boxchar<4>
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_28]] : (index) -> i64
// CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_31]], %[[VAL_26]] overflow<nsw> : i64
// CHECK:             %[[VAL_33:.*]] = arith.cmpi sge, %[[VAL_32]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_34:.*]] = arith.cmpi sle, %[[VAL_32]], %[[VAL_25]] : i64
// CHECK:             %[[VAL_35:.*]] = arith.andi %[[VAL_33]], %[[VAL_34]] : i1
// CHECK:             %[[VAL_36:.*]] = fir.if %[[VAL_35]] -> (!fir.boxchar<4>) {
// CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
// CHECK:               %[[VAL_38:.*]] = hlfir.designate %[[VAL_20]]#0 (%[[VAL_37]], %[[VAL_29]])  typeparams %[[VAL_10]] : (!fir.box<!fir.array<?x?x!fir.char<4,?>>>, index, index, i32) -> !fir.boxchar<4>
// CHECK:               fir.result %[[VAL_38]] : !fir.boxchar<4>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_30]] : !fir.boxchar<4>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_36]] : !fir.boxchar<4>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_27]] to %[[VAL_20]]#0 : !hlfir.expr<?x?x!fir.char<4,?>>, !fir.box<!fir.array<?x?x!fir.char<4,?>>>
// CHECK:           hlfir.destroy %[[VAL_27]] : !hlfir.expr<?x?x!fir.char<4,?>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the array always present boundary.
// ! CHARACTER with assumed length.
// subroutine eoshift13c(n, array, boundary)
//   integer :: n
//   character(*,4) :: array(n,n), boundary(:)
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift13c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<4> {fir.bindc_name = "array"}, %arg2: !fir.box<!fir.array<?x!fir.char<4,?>>> {fir.bindc_name = "boundary"}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift13cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFeoshift13cEboundary"} : (!fir.box<!fir.array<?x!fir.char<4,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<4,?>>>, !fir.box<!fir.array<?x!fir.char<4,?>>>)
  %3:2 = fir.unboxchar %arg1 : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
  %4 = fir.convert %3#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<4,?>>>
  %5 = fir.load %1#0 : !fir.ref<i32>
  %6 = fir.convert %5 : (i32) -> index
  %7 = arith.cmpi sgt, %6, %c0 : index
  %8 = arith.select %7, %6, %c0 : index
  %9 = fir.load %1#0 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = arith.cmpi sgt, %10, %c0 : index
  %12 = arith.select %11, %10, %c0 : index
  %13 = fir.shape %8, %12 : (index, index) -> !fir.shape<2>
  %14:2 = hlfir.declare %4(%13) typeparams %3#1 dummy_scope %0 {uniq_name = "_QFeoshift13cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<4,?>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<4,?>>>, !fir.ref<!fir.array<?x?x!fir.char<4,?>>>)
  %15 = hlfir.eoshift %14#0 %c2_i32 boundary %2#0 : (!fir.box<!fir.array<?x?x!fir.char<4,?>>>, i32, !fir.box<!fir.array<?x!fir.char<4,?>>>) -> !hlfir.expr<?x?x!fir.char<4,?>>
  hlfir.assign %15 to %14#0 : !hlfir.expr<?x?x!fir.char<4,?>>, !fir.box<!fir.array<?x?x!fir.char<4,?>>>
  hlfir.destroy %15 : !hlfir.expr<?x?x!fir.char<4,?>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift13c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<4> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.char<4,?>>> {fir.bindc_name = "boundary"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift13cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift13cEboundary"} : (!fir.box<!fir.array<?x!fir.char<4,?>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<4,?>>>, !fir.box<!fir.array<?x!fir.char<4,?>>>)
// CHECK:           %[[VAL_7:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<4,?>>>
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
// CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_3]] : index
// CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_3]] : index
// CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
// CHECK:           %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_3]] : index
// CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_14]], %[[VAL_3]] : index
// CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_12]], %[[VAL_16]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_17]]) typeparams %[[VAL_7]]#1 dummy_scope %[[VAL_4]] {uniq_name = "_QFeoshift13cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<4,?>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<4,?>>>, !fir.ref<!fir.array<?x?x!fir.char<4,?>>>)
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_12]] : (index) -> i64
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_2]] : (i32) -> i64
// CHECK:           %[[VAL_21:.*]] = hlfir.elemental %[[VAL_17]] typeparams %[[VAL_7]]#1 unordered : (!fir.shape<2>, index) -> !hlfir.expr<?x?x!fir.char<4,?>> {
// CHECK:           ^bb0(%[[VAL_22:.*]]: index, %[[VAL_23:.*]]: index):
// CHECK:             %[[VAL_24:.*]] = fir.box_elesize %[[VAL_6]]#1 : (!fir.box<!fir.array<?x!fir.char<4,?>>>) -> index
// CHECK:             %[[VAL_25:.*]] = arith.divsi %[[VAL_24]], %[[VAL_1]] : index
// CHECK:             %[[VAL_26:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_23]])  typeparams %[[VAL_25]] : (!fir.box<!fir.array<?x!fir.char<4,?>>>, index, index) -> !fir.boxchar<4>
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_22]] : (index) -> i64
// CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_27]], %[[VAL_20]] overflow<nsw> : i64
// CHECK:             %[[VAL_29:.*]] = arith.cmpi sge, %[[VAL_28]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_30:.*]] = arith.cmpi sle, %[[VAL_28]], %[[VAL_19]] : i64
// CHECK:             %[[VAL_31:.*]] = arith.andi %[[VAL_29]], %[[VAL_30]] : i1
// CHECK:             %[[VAL_32:.*]] = fir.if %[[VAL_31]] -> (!fir.boxchar<4>) {
// CHECK:               %[[VAL_33:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
// CHECK:               %[[VAL_34:.*]] = hlfir.designate %[[VAL_18]]#0 (%[[VAL_33]], %[[VAL_23]])  typeparams %[[VAL_7]]#1 : (!fir.box<!fir.array<?x?x!fir.char<4,?>>>, index, index, index) -> !fir.boxchar<4>
// CHECK:               fir.result %[[VAL_34]] : !fir.boxchar<4>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_26]] : !fir.boxchar<4>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_32]] : !fir.boxchar<4>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_21]] to %[[VAL_18]]#0 : !hlfir.expr<?x?x!fir.char<4,?>>, !fir.box<!fir.array<?x?x!fir.char<4,?>>>
// CHECK:           hlfir.destroy %[[VAL_21]] : !hlfir.expr<?x?x!fir.char<4,?>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the array optional boundary.
// ! CHARACTER with constant length.
// subroutine eoshift14c(n, array, boundary)
//   integer :: n
//   character(10,1) :: array(n,n)
//   character(10,1), optional :: boundary(n)
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift14c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "array"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "boundary", fir.optional}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift14cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,10>>>
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.cmpi sgt, %5, %c0 : index
  %7 = arith.select %6, %5, %c0 : index
  %8 = fir.load %1#0 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = arith.cmpi sgt, %9, %c0 : index
  %11 = arith.select %10, %9, %c0 : index
  %12 = fir.shape %7, %11 : (index, index) -> !fir.shape<2>
  %13:2 = hlfir.declare %3(%12) typeparams %c10 dummy_scope %0 {uniq_name = "_QFeoshift14cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<1,10>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x?x!fir.char<1,10>>>)
  %14:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %15 = fir.convert %14#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,10>>>
  %16 = fir.load %1#0 : !fir.ref<i32>
  %17 = fir.convert %16 : (i32) -> index
  %18 = arith.cmpi sgt, %17, %c0 : index
  %19 = arith.select %18, %17, %c0 : index
  %20 = fir.shape %19 : (index) -> !fir.shape<1>
  %21:2 = hlfir.declare %15(%20) typeparams %c10 dummy_scope %0 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift14cEboundary"} : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>)
  %22 = fir.is_present %21#0 : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> i1
  %23 = fir.shape %19 : (index) -> !fir.shape<1>
  %24 = fir.embox %21#1(%23) : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,10>>>
  %25 = fir.absent !fir.box<!fir.array<?x!fir.char<1,10>>>
  %26 = arith.select %22, %24, %25 : !fir.box<!fir.array<?x!fir.char<1,10>>>
  %27 = hlfir.eoshift %13#0 %c2_i32 boundary %26 : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, i32, !fir.box<!fir.array<?x!fir.char<1,10>>>) -> !hlfir.expr<?x?x!fir.char<1,10>>
  hlfir.assign %27 to %13#0 : !hlfir.expr<?x?x!fir.char<1,10>>, !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  hlfir.destroy %27 : !hlfir.expr<?x?x!fir.char<1,10>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift14c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "boundary", fir.optional}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_7]] {uniq_name = "_QFeoshift14cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_9:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
// CHECK:           %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_5]] : index
// CHECK:           %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_12]], %[[VAL_5]] : index
// CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
// CHECK:           %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_5]] : index
// CHECK:           %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_16]], %[[VAL_5]] : index
// CHECK:           %[[VAL_19:.*]] = fir.shape %[[VAL_14]], %[[VAL_18]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_19]]) typeparams %[[VAL_6]] dummy_scope %[[VAL_7]] {uniq_name = "_QFeoshift14cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<1,10>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x?x!fir.char<1,10>>>)
// CHECK:           %[[VAL_21:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,10>>>
// CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i32) -> index
// CHECK:           %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_24]], %[[VAL_5]] : index
// CHECK:           %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_24]], %[[VAL_5]] : index
// CHECK:           %[[VAL_27:.*]] = fir.shape %[[VAL_26]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_28:.*]]:2 = hlfir.declare %[[VAL_22]](%[[VAL_27]]) typeparams %[[VAL_6]] dummy_scope %[[VAL_7]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift14cEboundary"} : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>)
// CHECK:           %[[VAL_29:.*]] = fir.is_present %[[VAL_28]]#0 : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> i1
// CHECK:           %[[VAL_30:.*]] = fir.shape %[[VAL_26]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_31:.*]] = fir.embox %[[VAL_28]]#1(%[[VAL_30]]) : (!fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.array<?x!fir.char<1,10>>>
// CHECK:           %[[VAL_32:.*]] = fir.absent !fir.box<!fir.array<?x!fir.char<1,10>>>
// CHECK:           %[[VAL_33:.*]] = arith.select %[[VAL_29]], %[[VAL_31]], %[[VAL_32]] : !fir.box<!fir.array<?x!fir.char<1,10>>>
// CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_14]] : (index) -> i64
// CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
// CHECK:           %[[VAL_36:.*]] = fir.alloca !fir.char<1,0> {bindc_name = ".chrtmp"}
// CHECK:           %[[VAL_37:.*]] = fir.emboxchar %[[VAL_36]], %[[VAL_5]] : (!fir.ref<!fir.char<1,0>>, index) -> !fir.boxchar<1>
// CHECK:           %[[VAL_38:.*]] = fir.is_present %[[VAL_33]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>) -> i1
// CHECK:           %[[VAL_39:.*]] = arith.select %[[VAL_38]], %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           %[[VAL_40:.*]] = hlfir.elemental %[[VAL_19]] typeparams %[[VAL_6]] unordered : (!fir.shape<2>, index) -> !hlfir.expr<?x?x!fir.char<1,10>> {
// CHECK:           ^bb0(%[[VAL_41:.*]]: index, %[[VAL_42:.*]]: index):
// CHECK:             %[[VAL_43:.*]] = fir.if %[[VAL_39]] -> (!fir.boxchar<1>) {
// CHECK:               fir.result %[[VAL_37]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               %[[VAL_44:.*]]:3 = fir.box_dims %[[VAL_33]], %[[VAL_5]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_45:.*]] = arith.subi %[[VAL_44]]#0, %[[VAL_1]] overflow<nsw> : index
// CHECK:               %[[VAL_46:.*]] = arith.addi %[[VAL_42]], %[[VAL_45]] overflow<nsw> : index
// CHECK:               %[[VAL_47:.*]] = hlfir.designate %[[VAL_33]] (%[[VAL_46]])  typeparams %[[VAL_6]] : (!fir.box<!fir.array<?x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
// CHECK:               %[[VAL_48:.*]] = fir.emboxchar %[[VAL_47]], %[[VAL_6]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_48]] : !fir.boxchar<1>
// CHECK:             }
// CHECK:             %[[VAL_49:.*]] = fir.convert %[[VAL_41]] : (index) -> i64
// CHECK:             %[[VAL_50:.*]] = arith.addi %[[VAL_49]], %[[VAL_35]] overflow<nsw> : i64
// CHECK:             %[[VAL_51:.*]] = arith.cmpi sge, %[[VAL_50]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_52:.*]] = arith.cmpi sle, %[[VAL_50]], %[[VAL_34]] : i64
// CHECK:             %[[VAL_53:.*]] = arith.andi %[[VAL_51]], %[[VAL_52]] : i1
// CHECK:             %[[VAL_54:.*]] = fir.if %[[VAL_53]] -> (!fir.boxchar<1>) {
// CHECK:               %[[VAL_55:.*]] = fir.convert %[[VAL_50]] : (i64) -> index
// CHECK:               %[[VAL_56:.*]] = hlfir.designate %[[VAL_20]]#0 (%[[VAL_55]], %[[VAL_42]])  typeparams %[[VAL_6]] : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index, index, index) -> !fir.ref<!fir.char<1,10>>
// CHECK:               %[[VAL_57:.*]] = fir.emboxchar %[[VAL_56]], %[[VAL_6]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_57]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_43]] : !fir.boxchar<1>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_54]] : !fir.boxchar<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_40]] to %[[VAL_20]]#0 : !hlfir.expr<?x?x!fir.char<1,10>>, !fir.box<!fir.array<?x?x!fir.char<1,10>>>
// CHECK:           hlfir.destroy %[[VAL_40]] : !hlfir.expr<?x?x!fir.char<1,10>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the array optional boundary.
// ! CHARACTER with variable length.
// subroutine eoshift15c(n, array, boundary)
//   integer :: n
//   character(n,1) :: array(n,n)
//   character(n,1), optional :: boundary(n)
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift15c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "array"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "boundary", fir.optional}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift15cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = arith.cmpi sgt, %4, %c0_i32 : i32
  %6 = arith.select %5, %4, %c0_i32 : i32
  %7 = fir.load %1#0 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = arith.cmpi sgt, %8, %c0 : index
  %10 = arith.select %9, %8, %c0 : index
  %11 = fir.load %1#0 : !fir.ref<i32>
  %12 = fir.convert %11 : (i32) -> index
  %13 = arith.cmpi sgt, %12, %c0 : index
  %14 = arith.select %13, %12, %c0 : index
  %15 = fir.shape %10, %14 : (index, index) -> !fir.shape<2>
  %16:2 = hlfir.declare %3(%15) typeparams %6 dummy_scope %0 {uniq_name = "_QFeoshift15cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x?x!fir.char<1,?>>>)
  %17:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %18 = fir.convert %17#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  %19 = fir.load %1#0 : !fir.ref<i32>
  %20 = arith.cmpi sgt, %19, %c0_i32 : i32
  %21 = arith.select %20, %19, %c0_i32 : i32
  %22 = fir.load %1#0 : !fir.ref<i32>
  %23 = fir.convert %22 : (i32) -> index
  %24 = arith.cmpi sgt, %23, %c0 : index
  %25 = arith.select %24, %23, %c0 : index
  %26 = fir.shape %25 : (index) -> !fir.shape<1>
  %27:2 = hlfir.declare %18(%26) typeparams %21 dummy_scope %0 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift15cEboundary"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
  %28 = fir.is_present %27#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
  %29 = fir.shape %25 : (index) -> !fir.shape<1>
  %30 = fir.embox %27#1(%29) typeparams %21 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  %31 = fir.absent !fir.box<!fir.array<?x!fir.char<1,?>>>
  %32 = arith.select %28, %30, %31 : !fir.box<!fir.array<?x!fir.char<1,?>>>
  %33 = hlfir.eoshift %16#0 %c2_i32 boundary %32 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32, !fir.box<!fir.array<?x!fir.char<1,?>>>) -> !hlfir.expr<?x?x!fir.char<1,?>>
  hlfir.assign %33 to %16#0 : !hlfir.expr<?x?x!fir.char<1,?>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  hlfir.destroy %33 : !hlfir.expr<?x?x!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift15c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "boundary", fir.optional}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_7]] {uniq_name = "_QFeoshift15cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_9:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_11]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
// CHECK:           %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_5]] : index
// CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_15]], %[[VAL_5]] : index
// CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
// CHECK:           %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_5]] : index
// CHECK:           %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_19]], %[[VAL_5]] : index
// CHECK:           %[[VAL_22:.*]] = fir.shape %[[VAL_17]], %[[VAL_21]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_23:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_22]]) typeparams %[[VAL_13]] dummy_scope %[[VAL_7]] {uniq_name = "_QFeoshift15cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x?x!fir.char<1,?>>>)
// CHECK:           %[[VAL_24:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_27:.*]] = arith.cmpi sgt, %[[VAL_26]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_28:.*]] = arith.select %[[VAL_27]], %[[VAL_26]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> index
// CHECK:           %[[VAL_31:.*]] = arith.cmpi sgt, %[[VAL_30]], %[[VAL_5]] : index
// CHECK:           %[[VAL_32:.*]] = arith.select %[[VAL_31]], %[[VAL_30]], %[[VAL_5]] : index
// CHECK:           %[[VAL_33:.*]] = fir.shape %[[VAL_32]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_34:.*]]:2 = hlfir.declare %[[VAL_25]](%[[VAL_33]]) typeparams %[[VAL_28]] dummy_scope %[[VAL_7]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift15cEboundary"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
// CHECK:           %[[VAL_35:.*]] = fir.is_present %[[VAL_34]]#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_36:.*]] = fir.shape %[[VAL_32]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_37:.*]] = fir.embox %[[VAL_34]]#1(%[[VAL_36]]) typeparams %[[VAL_28]] : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_38:.*]] = fir.absent !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_39:.*]] = arith.select %[[VAL_35]], %[[VAL_37]], %[[VAL_38]] : !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
// CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
// CHECK:           %[[VAL_42:.*]] = fir.alloca !fir.char<1,0> {bindc_name = ".chrtmp"}
// CHECK:           %[[VAL_43:.*]] = fir.emboxchar %[[VAL_42]], %[[VAL_5]] : (!fir.ref<!fir.char<1,0>>, index) -> !fir.boxchar<1>
// CHECK:           %[[VAL_44:.*]] = fir.is_present %[[VAL_39]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_45:.*]] = arith.select %[[VAL_44]], %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           %[[VAL_46:.*]] = hlfir.elemental %[[VAL_22]] typeparams %[[VAL_13]] unordered : (!fir.shape<2>, i32) -> !hlfir.expr<?x?x!fir.char<1,?>> {
// CHECK:           ^bb0(%[[VAL_47:.*]]: index, %[[VAL_48:.*]]: index):
// CHECK:             %[[VAL_49:.*]] = fir.if %[[VAL_45]] -> (!fir.boxchar<1>) {
// CHECK:               fir.result %[[VAL_43]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               %[[VAL_50:.*]] = fir.box_elesize %[[VAL_39]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
// CHECK:               %[[VAL_51:.*]]:3 = fir.box_dims %[[VAL_39]], %[[VAL_5]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_52:.*]] = arith.subi %[[VAL_51]]#0, %[[VAL_1]] overflow<nsw> : index
// CHECK:               %[[VAL_53:.*]] = arith.addi %[[VAL_48]], %[[VAL_52]] overflow<nsw> : index
// CHECK:               %[[VAL_54:.*]] = hlfir.designate %[[VAL_39]] (%[[VAL_53]])  typeparams %[[VAL_50]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_54]] : !fir.boxchar<1>
// CHECK:             }
// CHECK:             %[[VAL_55:.*]] = fir.convert %[[VAL_47]] : (index) -> i64
// CHECK:             %[[VAL_56:.*]] = arith.addi %[[VAL_55]], %[[VAL_41]] overflow<nsw> : i64
// CHECK:             %[[VAL_57:.*]] = arith.cmpi sge, %[[VAL_56]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_58:.*]] = arith.cmpi sle, %[[VAL_56]], %[[VAL_40]] : i64
// CHECK:             %[[VAL_59:.*]] = arith.andi %[[VAL_57]], %[[VAL_58]] : i1
// CHECK:             %[[VAL_60:.*]] = fir.if %[[VAL_59]] -> (!fir.boxchar<1>) {
// CHECK:               %[[VAL_61:.*]] = fir.convert %[[VAL_56]] : (i64) -> index
// CHECK:               %[[VAL_62:.*]] = hlfir.designate %[[VAL_23]]#0 (%[[VAL_61]], %[[VAL_48]])  typeparams %[[VAL_13]] : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index, index, i32) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_62]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_49]] : !fir.boxchar<1>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_60]] : !fir.boxchar<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_46]] to %[[VAL_23]]#0 : !hlfir.expr<?x?x!fir.char<1,?>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           hlfir.destroy %[[VAL_46]] : !hlfir.expr<?x?x!fir.char<1,?>>
// CHECK:           return
// CHECK:         }

// ! Test contiguous 1D array with the array optional boundary.
// ! CHARACTER with assumed length.
// subroutine eoshift16c(n, array, boundary)
//   integer :: n
//   character(*,1) :: array(n,n)
//   character(*,1), optional :: boundary(n)
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift16c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "array"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "boundary", fir.optional}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift16cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.cmpi sgt, %5, %c0 : index
  %7 = arith.select %6, %5, %c0 : index
  %8 = fir.load %1#0 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = arith.cmpi sgt, %9, %c0 : index
  %11 = arith.select %10, %9, %c0 : index
  %12 = fir.shape %7, %11 : (index, index) -> !fir.shape<2>
  %13:2 = hlfir.declare %3(%12) typeparams %2#1 dummy_scope %0 {uniq_name = "_QFeoshift16cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x?x!fir.char<1,?>>>)
  %14:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %15 = fir.convert %14#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  %16 = fir.load %1#0 : !fir.ref<i32>
  %17 = fir.convert %16 : (i32) -> index
  %18 = arith.cmpi sgt, %17, %c0 : index
  %19 = arith.select %18, %17, %c0 : index
  %20 = fir.shape %19 : (index) -> !fir.shape<1>
  %21:2 = hlfir.declare %15(%20) typeparams %14#1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift16cEboundary"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
  %22 = fir.is_present %21#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
  %23 = fir.shape %19 : (index) -> !fir.shape<1>
  %24 = fir.embox %21#1(%23) typeparams %14#1 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  %25 = fir.absent !fir.box<!fir.array<?x!fir.char<1,?>>>
  %26 = arith.select %22, %24, %25 : !fir.box<!fir.array<?x!fir.char<1,?>>>
  %27 = hlfir.eoshift %13#0 %c2_i32 boundary %26 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, i32, !fir.box<!fir.array<?x!fir.char<1,?>>>) -> !hlfir.expr<?x?x!fir.char<1,?>>
  hlfir.assign %27 to %13#0 : !hlfir.expr<?x?x!fir.char<1,?>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>
  hlfir.destroy %27 : !hlfir.expr<?x?x!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift16c(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "boundary", fir.optional}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_6]] {uniq_name = "_QFeoshift16cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_8:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
// CHECK:           %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_5]] : index
// CHECK:           %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_11]], %[[VAL_5]] : index
// CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
// CHECK:           %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_5]] : index
// CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_15]], %[[VAL_5]] : index
// CHECK:           %[[VAL_18:.*]] = fir.shape %[[VAL_13]], %[[VAL_17]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_18]]) typeparams %[[VAL_8]]#1 dummy_scope %[[VAL_6]] {uniq_name = "_QFeoshift16cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x?x!fir.char<1,?>>>)
// CHECK:           %[[VAL_20:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_22:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> index
// CHECK:           %[[VAL_24:.*]] = arith.cmpi sgt, %[[VAL_23]], %[[VAL_5]] : index
// CHECK:           %[[VAL_25:.*]] = arith.select %[[VAL_24]], %[[VAL_23]], %[[VAL_5]] : index
// CHECK:           %[[VAL_26:.*]] = fir.shape %[[VAL_25]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_27:.*]]:2 = hlfir.declare %[[VAL_21]](%[[VAL_26]]) typeparams %[[VAL_20]]#1 dummy_scope %[[VAL_6]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift16cEboundary"} : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>)
// CHECK:           %[[VAL_28:.*]] = fir.is_present %[[VAL_27]]#0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_29:.*]] = fir.shape %[[VAL_25]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_30:.*]] = fir.embox %[[VAL_27]]#1(%[[VAL_29]]) typeparams %[[VAL_20]]#1 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_31:.*]] = fir.absent !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_32:.*]] = arith.select %[[VAL_28]], %[[VAL_30]], %[[VAL_31]] : !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_13]] : (index) -> i64
// CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
// CHECK:           %[[VAL_35:.*]] = fir.alloca !fir.char<1,0> {bindc_name = ".chrtmp"}
// CHECK:           %[[VAL_36:.*]] = fir.emboxchar %[[VAL_35]], %[[VAL_5]] : (!fir.ref<!fir.char<1,0>>, index) -> !fir.boxchar<1>
// CHECK:           %[[VAL_37:.*]] = fir.is_present %[[VAL_32]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> i1
// CHECK:           %[[VAL_38:.*]] = arith.select %[[VAL_37]], %[[VAL_2]], %[[VAL_3]] : i1
// CHECK:           %[[VAL_39:.*]] = hlfir.elemental %[[VAL_18]] typeparams %[[VAL_8]]#1 unordered : (!fir.shape<2>, index) -> !hlfir.expr<?x?x!fir.char<1,?>> {
// CHECK:           ^bb0(%[[VAL_40:.*]]: index, %[[VAL_41:.*]]: index):
// CHECK:             %[[VAL_42:.*]] = fir.if %[[VAL_38]] -> (!fir.boxchar<1>) {
// CHECK:               fir.result %[[VAL_36]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               %[[VAL_43:.*]] = fir.box_elesize %[[VAL_32]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
// CHECK:               %[[VAL_44:.*]]:3 = fir.box_dims %[[VAL_32]], %[[VAL_5]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
// CHECK:               %[[VAL_45:.*]] = arith.subi %[[VAL_44]]#0, %[[VAL_1]] overflow<nsw> : index
// CHECK:               %[[VAL_46:.*]] = arith.addi %[[VAL_41]], %[[VAL_45]] overflow<nsw> : index
// CHECK:               %[[VAL_47:.*]] = hlfir.designate %[[VAL_32]] (%[[VAL_46]])  typeparams %[[VAL_43]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_47]] : !fir.boxchar<1>
// CHECK:             }
// CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_40]] : (index) -> i64
// CHECK:             %[[VAL_49:.*]] = arith.addi %[[VAL_48]], %[[VAL_34]] overflow<nsw> : i64
// CHECK:             %[[VAL_50:.*]] = arith.cmpi sge, %[[VAL_49]], %[[VAL_0]] : i64
// CHECK:             %[[VAL_51:.*]] = arith.cmpi sle, %[[VAL_49]], %[[VAL_33]] : i64
// CHECK:             %[[VAL_52:.*]] = arith.andi %[[VAL_50]], %[[VAL_51]] : i1
// CHECK:             %[[VAL_53:.*]] = fir.if %[[VAL_52]] -> (!fir.boxchar<1>) {
// CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_49]] : (i64) -> index
// CHECK:               %[[VAL_55:.*]] = hlfir.designate %[[VAL_19]]#0 (%[[VAL_54]], %[[VAL_41]])  typeparams %[[VAL_8]]#1 : (!fir.box<!fir.array<?x?x!fir.char<1,?>>>, index, index, index) -> !fir.boxchar<1>
// CHECK:               fir.result %[[VAL_55]] : !fir.boxchar<1>
// CHECK:             } else {
// CHECK:               fir.result %[[VAL_42]] : !fir.boxchar<1>
// CHECK:             }
// CHECK:             hlfir.yield_element %[[VAL_53]] : !fir.boxchar<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_39]] to %[[VAL_19]]#0 : !hlfir.expr<?x?x!fir.char<1,?>>, !fir.box<!fir.array<?x?x!fir.char<1,?>>>
// CHECK:           hlfir.destroy %[[VAL_39]] : !hlfir.expr<?x?x!fir.char<1,?>>
// CHECK:           return
// CHECK:         }

// ! TODO: ARRAY or/and BOUNDARY are expressions of CHARACTER type.
// ! Test contiguous 1D array with the array expression boundary.
// ! CHARACTER with constant length.
// subroutine eoshift17c(n, array)
//   interface
//      function charc_boundary(n)
//        integer :: n
//        character(10,1) :: charc_boundary(n)
//      end function
//   end interface
//   integer :: n
//   character(10,1) :: array(n,n)
//   array = EOSHIFT(array//array, 2, charc_boundary(n))
// end subroutine
func.func @_QPeoshift17c(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "array"}) {
  %c20 = arith.constant 20 : index
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift17cEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3 = fir.convert %2#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x?x!fir.char<1,10>>>
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %6 = arith.cmpi sgt, %5, %c0 : index
  %7 = arith.select %6, %5, %c0 : index
  %8 = fir.load %1#0 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> index
  %10 = arith.cmpi sgt, %9, %c0 : index
  %11 = arith.select %10, %9, %c0 : index
  %12 = fir.shape %7, %11 : (index, index) -> !fir.shape<2>
  %13:2 = hlfir.declare %3(%12) typeparams %c10 dummy_scope %0 {uniq_name = "_QFeoshift17cEarray"} : (!fir.ref<!fir.array<?x?x!fir.char<1,10>>>, !fir.shape<2>, index, !fir.dscope) -> (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, !fir.ref<!fir.array<?x?x!fir.char<1,10>>>)
  %14 = hlfir.elemental %12 typeparams %c20 unordered : (!fir.shape<2>, index) -> !hlfir.expr<?x?x!fir.char<1,?>> {
  ^bb0(%arg2: index, %arg3: index):
    %23 = hlfir.designate %13#0 (%arg2, %arg3)  typeparams %c10 : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index, index, index) -> !fir.ref<!fir.char<1,10>>
    %24 = hlfir.designate %13#0 (%arg2, %arg3)  typeparams %c10 : (!fir.box<!fir.array<?x?x!fir.char<1,10>>>, index, index, index) -> !fir.ref<!fir.char<1,10>>
    %25 = hlfir.concat %23, %24 len %c20 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>, index) -> !hlfir.expr<!fir.char<1,20>>
    hlfir.yield_element %25 : !hlfir.expr<!fir.char<1,20>>
  }
  %15:2 = hlfir.declare %1#0 {uniq_name = "_QFeoshift17cFcharc_boundaryEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %16 = fir.load %15#0 : !fir.ref<i32>
  %17 = fir.convert %16 : (i32) -> index
  %18 = arith.cmpi sgt, %17, %c0 : index
  %19 = arith.select %18, %17, %c0 : index
  %20 = fir.shape %19 : (index) -> !fir.shape<1>
  %21 = hlfir.eval_in_mem shape %20 typeparams %c10 : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,10>> {
  ^bb0(%arg2: !fir.ref<!fir.array<?x!fir.char<1,10>>>):
    %23 = fir.call @_QPcharc_boundary(%1#0) fastmath<contract> : (!fir.ref<i32>) -> !fir.array<?x!fir.char<1,10>>
    fir.save_result %23 to %arg2(%20) typeparams %c10 : !fir.array<?x!fir.char<1,10>>, !fir.ref<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>, index
  }
  %22 = hlfir.eoshift %14 %c2_i32 boundary %21 : (!hlfir.expr<?x?x!fir.char<1,?>>, i32, !hlfir.expr<?x!fir.char<1,10>>) -> !hlfir.expr<?x?x!fir.char<1,20>>
  hlfir.assign %22 to %13#0 : !hlfir.expr<?x?x!fir.char<1,20>>, !fir.box<!fir.array<?x?x!fir.char<1,10>>>
  hlfir.destroy %22 : !hlfir.expr<?x?x!fir.char<1,20>>
  hlfir.destroy %21 : !hlfir.expr<?x!fir.char<1,10>>
  hlfir.destroy %14 : !hlfir.expr<?x?x!fir.char<1,?>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift17c(
// CHECK: hlfir.eoshift

// ! Tests for derived types.

// ! TODO: selecting between !fir.ref<!fir.type<>> and !fir.box<!fir.type<>>
// ! is not implemented.
// ! Test contiguous 1D array with the scalar optional boundary.
// subroutine eoshift1d(n, array, boundary)
//   use eoshift_types
//   integer :: n
//   type(t) :: array(n)
//   type(t), optional :: boundary
//   array = EOSHIFT(array, 2, boundary)
// end subroutine
func.func @_QPeoshift1d(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<!fir.array<?x!fir.type<_QMeoshift_typesTt>>> {fir.bindc_name = "array"}, %arg2: !fir.ref<!fir.type<_QMeoshift_typesTt>> {fir.bindc_name = "boundary", fir.optional}) {
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift1dEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 dummy_scope %0 {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFeoshift1dEboundary"} : (!fir.ref<!fir.type<_QMeoshift_typesTt>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMeoshift_typesTt>>, !fir.ref<!fir.type<_QMeoshift_typesTt>>)
  %3 = fir.load %1#0 : !fir.ref<i32>
  %4 = fir.convert %3 : (i32) -> index
  %5 = arith.cmpi sgt, %4, %c0 : index
  %6 = arith.select %5, %4, %c0 : index
  %7 = fir.shape %6 : (index) -> !fir.shape<1>
  %8:2 = hlfir.declare %arg1(%7) dummy_scope %0 {uniq_name = "_QFeoshift1dEarray"} : (!fir.ref<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>, !fir.ref<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>)
  %9 = fir.is_present %2#0 : (!fir.ref<!fir.type<_QMeoshift_typesTt>>) -> i1
  %10 = fir.embox %2#0 : (!fir.ref<!fir.type<_QMeoshift_typesTt>>) -> !fir.box<!fir.type<_QMeoshift_typesTt>>
  %11 = fir.absent !fir.box<!fir.type<_QMeoshift_typesTt>>
  %12 = arith.select %9, %10, %11 : !fir.box<!fir.type<_QMeoshift_typesTt>>
  %13 = hlfir.eoshift %8#0 %c2_i32 boundary %12 : (!fir.box<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>, i32, !fir.box<!fir.type<_QMeoshift_typesTt>>) -> !hlfir.expr<?x!fir.type<_QMeoshift_typesTt>>
  hlfir.assign %13 to %8#0 : !hlfir.expr<?x!fir.type<_QMeoshift_typesTt>>, !fir.box<!fir.array<?x!fir.type<_QMeoshift_typesTt>>>
  hlfir.destroy %13 : !hlfir.expr<?x!fir.type<_QMeoshift_typesTt>>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift1d(
// CHECK: hlfir.eoshift
