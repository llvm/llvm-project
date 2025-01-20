! RUN: bbc -emit-hlfir -o - %s -I nowhere | FileCheck %s

subroutine simple_test()
  implicit none
  interface
    function array_func()
      real :: array_func(10)
    end function
  end interface
  real :: x(10)
  x = array_func()
end subroutine

subroutine arg_test(n)
  implicit none
  interface
    function array_func_2(n)
      integer(8) :: n
      real :: array_func_2(n)
    end function
  end interface
  integer(8) :: n
  real :: x(n)
  x = array_func_2(n)
end subroutine

module type_defs
  interface
    function array_func()
      real :: array_func(10)
    end function
  end interface
  type t
    contains
    procedure, nopass :: array_func => array_func
  end type
end module

subroutine dispatch_test(x, a)
  use type_defs, only : t
  implicit none
  real :: x(10)
  class(t) :: a
  x = a%array_func()
end subroutine

! CHECK-LABEL:   func.func @_QPsimple_test() {
! CHECK:           %[[VAL_0:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<10xf32> {bindc_name = "x", uniq_name = "_QFsimple_testEx"}
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) {uniq_name = "_QFsimple_testEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : i64
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_6:.*]] = arith.subi %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_10]] : index
! CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_9]], %[[VAL_10]] : index
! CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_14:.*]] = hlfir.eval_in_mem shape %[[VAL_13]] : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
! CHECK:           ^bb0(%[[VAL_15:.*]]: !fir.ref<!fir.array<10xf32>>):
! CHECK:             %[[VAL_16:.*]] = fir.call @_QParray_func() fastmath<contract> : () -> !fir.array<10xf32>
! CHECK:             fir.save_result %[[VAL_16]] to %[[VAL_15]](%[[VAL_13]]) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_14]] to %[[VAL_3]]#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
! CHECK:           hlfir.destroy %[[VAL_14]] : !hlfir.expr<10xf32>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QParg_test(
! CHECK-SAME:                           %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFarg_testEn"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_4]], %[[VAL_5]] : index
! CHECK:           %[[VAL_7:.*]] = arith.select %[[VAL_6]], %[[VAL_4]], %[[VAL_5]] : index
! CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.array<?xf32>, %[[VAL_7]] {bindc_name = "x", uniq_name = "_QFarg_testEx"}
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_9]]) {uniq_name = "_QFarg_testEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_2]]#1 {uniq_name = "_QFarg_testFarray_func_2En"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_13:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_13]] : i64
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_14]], %[[VAL_15]] : i64
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:           %[[VAL_18:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_17]], %[[VAL_18]] : index
! CHECK:           %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_17]], %[[VAL_18]] : index
! CHECK:           %[[VAL_21:.*]] = fir.shape %[[VAL_20]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_22:.*]] = hlfir.eval_in_mem shape %[[VAL_21]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:           ^bb0(%[[VAL_23:.*]]: !fir.ref<!fir.array<?xf32>>):
! CHECK:             %[[VAL_24:.*]] = fir.call @_QParray_func_2(%[[VAL_2]]#1) fastmath<contract> : (!fir.ref<i64>) -> !fir.array<?xf32>
! CHECK:             fir.save_result %[[VAL_24]] to %[[VAL_23]](%[[VAL_21]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_22]] to %[[VAL_10]]#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
! CHECK:           hlfir.destroy %[[VAL_22]] : !hlfir.expr<?xf32>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPdispatch_test(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "x"},
! CHECK-SAME:                                %[[VAL_1:.*]]: !fir.class<!fir.type<_QMtype_defsTt>> {fir.bindc_name = "a"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_2]] {uniq_name = "_QFdispatch_testEa"} : (!fir.class<!fir.type<_QMtype_defsTt>>, !fir.dscope) -> (!fir.class<!fir.type<_QMtype_defsTt>>, !fir.class<!fir.type<_QMtype_defsTt>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_5]]) dummy_scope %[[VAL_2]] {uniq_name = "_QFdispatch_testEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 10 : i64
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_7]], %[[VAL_8]] : i64
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_13]] : index
! CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_12]], %[[VAL_13]] : index
! CHECK:           %[[VAL_16:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_17:.*]] = hlfir.eval_in_mem shape %[[VAL_16]] : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
! CHECK:           ^bb0(%[[VAL_18:.*]]: !fir.ref<!fir.array<10xf32>>):
! CHECK:             %[[VAL_19:.*]] = fir.dispatch "array_func"(%[[VAL_3]]#1 : !fir.class<!fir.type<_QMtype_defsTt>>) -> !fir.array<10xf32>
! CHECK:             fir.save_result %[[VAL_19]] to %[[VAL_18]](%[[VAL_16]]) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_17]] to %[[VAL_6]]#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
! CHECK:           hlfir.destroy %[[VAL_17]] : !hlfir.expr<10xf32>
! CHECK:           return
! CHECK:         }
