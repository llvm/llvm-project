! Test lowering of array constructors as hlfir.elemental.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_as_simple_elemental(n)
  integer :: n
  call takes_int([(n+i, i=1,4)])
end subroutine
! CHECK-LABEL:   func.func @_QPtest_as_simple_elemental(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_as_simple_elementalEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_9:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<1>) -> !hlfir.expr<4xi32> {
! CHECK:           ^bb0(%[[VAL_10:.*]]: index):
! CHECK:             %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_8]] : index
! CHECK:             %[[VAL_12:.*]] = arith.muli %[[VAL_11]], %[[VAL_7]] : index
! CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_5]], %[[VAL_12]] : index
! CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (index) -> i64
! CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> i32
! CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_14]], %[[VAL_16]] : i32
! CHECK:             hlfir.yield_element %[[VAL_17]] : i32
! CHECK:           }
! CHECK:           %[[VAL_18:.*]]:3 = hlfir.associate %[[VAL_19:.*]](%[[VAL_3]]) {adapt.valuebyref} : (!hlfir.expr<4xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>, i1)
! CHECK:           fir.call @_QPtakes_int(%[[VAL_18]]#1) fastmath<contract> : (!fir.ref<!fir.array<4xi32>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_18]]#1, %[[VAL_18]]#2 : !fir.ref<!fir.array<4xi32>>, i1
! CHECK:           hlfir.destroy %[[VAL_19]] : !hlfir.expr<4xi32>
! CHECK:           return
! CHECK:         }

subroutine test_as_strided_elemental(lb, ub, stride)
  integer(8) :: lb, ub, stride
  call takes_int([(i, i=lb,ub,stride)])
end subroutine
! CHECK-LABEL:   func.func @_QPtest_as_strided_elemental(
! CHECK-SAME:                                            %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "lb"},
! CHECK-SAME:                                            %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "ub"},
! CHECK-SAME:                                            %[[VAL_2:.*]]: !fir.ref<i64> {fir.bindc_name = "stride"}) {
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_as_strided_elementalElb"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest_as_strided_elementalEstride"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest_as_strided_elementalEub"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_7]], %[[VAL_8]] : i64
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:           %[[VAL_12:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_13:.*]] = arith.divsi %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:           %[[VAL_14:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_14]] : i64
! CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_13]], %[[VAL_14]] : i64
! CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_6]], %[[VAL_16]] : i64
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
! CHECK:           %[[VAL_19:.*]] = fir.shape %[[VAL_18]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i64>
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_25:.*]] = hlfir.elemental %[[VAL_19]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:           ^bb0(%[[VAL_26:.*]]: index):
! CHECK:             %[[VAL_27:.*]] = arith.subi %[[VAL_26]], %[[VAL_24]] : index
! CHECK:             %[[VAL_28:.*]] = arith.muli %[[VAL_27]], %[[VAL_23]] : index
! CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_21]], %[[VAL_28]] : index
! CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (index) -> i64
! CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> i32
! CHECK:             hlfir.yield_element %[[VAL_31]] : i32
! CHECK:           }
! CHECK:           %[[VAL_32:.*]]:3 = hlfir.associate %[[VAL_33:.*]](%[[VAL_19]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_32]]#1 : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<4xi32>>
! CHECK:           fir.call @_QPtakes_int(%[[VAL_34]]) fastmath<contract> : (!fir.ref<!fir.array<4xi32>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_32]]#1, %[[VAL_32]]#2 : !fir.ref<!fir.array<?xi32>>, i1
! CHECK:           hlfir.destroy %[[VAL_33]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }

subroutine test_as_elemental_with_pure_call(n)
  interface
    integer pure function foo(i)
      integer, value :: i
    end function
  end interface
  integer :: n
  call takes_int([(foo(i), i=1,4)])
end subroutine
! CHECK-LABEL:   func.func @_QPtest_as_elemental_with_pure_call(
! CHECK-SAME:                                                   %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_as_elemental_with_pure_callEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_9:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<1>) -> !hlfir.expr<4xi32> {
! CHECK:           ^bb0(%[[VAL_10:.*]]: index):
! CHECK:             %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_8]] : index
! CHECK:             %[[VAL_12:.*]] = arith.muli %[[VAL_11]], %[[VAL_7]] : index
! CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_5]], %[[VAL_12]] : index
! CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (index) -> i64
! CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> i32
! CHECK:             %[[VAL_16:.*]] = fir.call @_QPfoo(%[[VAL_15]]) fastmath<contract> : (i32) -> i32
! CHECK:             hlfir.yield_element %[[VAL_16]] : i32
! CHECK:           }
! CHECK:           %[[VAL_17:.*]]:3 = hlfir.associate %[[VAL_18:.*]](%[[VAL_3]]) {adapt.valuebyref} : (!hlfir.expr<4xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>, i1)
! CHECK:           fir.call @_QPtakes_int(%[[VAL_17]]#1) fastmath<contract> : (!fir.ref<!fir.array<4xi32>>) -> ()
! CHECK:           hlfir.end_associate %[[VAL_17]]#1, %[[VAL_17]]#2 : !fir.ref<!fir.array<4xi32>>, i1
! CHECK:           hlfir.destroy %[[VAL_18]] : !hlfir.expr<4xi32>
! CHECK:           return
! CHECK:         }

! CHECK-LABEL: func.func @_QPtest_with_impure_call(
subroutine test_with_impure_call(n)
  interface
    integer function impure_foo(i)
      integer, value :: i
    end function
  end interface
  integer :: n
  call takes_int([(impure_foo(i), i=1,4)])
end subroutine
! CHECK-NOT: hlfir.elemental
! CHECK:  return

! Test that the hlfir.expr result of the outer intrinsic call
! is not destructed.
subroutine test_hlfir_expr_result_destruction
  character(4) :: a(21)
  a = (/ (repeat(repeat(char(i),2),2),i=1,n) /)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_hlfir_expr_result_destruction() {
! CHECK:           %[[VAL_36:.*]] = hlfir.elemental %{{.*}} typeparams %{{.*}} unordered : (!fir.shape<1>, index) -> !hlfir.expr<?x!fir.char<1,?>> {
! CHECK:             %[[VAL_48:.*]] = hlfir.as_expr %{{.*}} move %{{.*}} : (!fir.ref<!fir.char<1>>, i1) -> !hlfir.expr<!fir.char<1>>
! CHECK:             %[[VAL_51:.*]]:3 = hlfir.associate %[[VAL_48]] typeparams %{{.*}} {adapt.valuebyref} : (!hlfir.expr<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>, i1)
! CHECK:             %[[VAL_64:.*]]:2 = hlfir.declare %{{.*}} typeparams %{{.*}} {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,2>>, index) -> (!fir.heap<!fir.char<1,2>>, !fir.heap<!fir.char<1,2>>)
! CHECK:             %[[VAL_66:.*]] = hlfir.as_expr %[[VAL_64]]#0 move %{{.*}} : (!fir.heap<!fir.char<1,2>>, i1) -> !hlfir.expr<!fir.char<1,2>>
! CHECK:             %[[VAL_68:.*]]:3 = hlfir.associate %[[VAL_66]] typeparams %{{.*}} {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>, i1)
! CHECK:             %[[VAL_81:.*]]:2 = hlfir.declare %{{.*}} typeparams %{{.*}} {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,4>>, index) -> (!fir.heap<!fir.char<1,4>>, !fir.heap<!fir.char<1,4>>)
! CHECK:             %[[VAL_83:.*]] = hlfir.as_expr %[[VAL_81]]#0 move %{{.*}} : (!fir.heap<!fir.char<1,4>>, i1) -> !hlfir.expr<!fir.char<1,4>>
! CHECK-NOT:         hlfir.destroy %[[VAL_83]]
! CHECK:             hlfir.end_associate %[[VAL_68]]#1, %[[VAL_68]]#2 : !fir.ref<!fir.char<1,2>>, i1
! CHECK-NOT:         hlfir.destroy %[[VAL_83]]
! CHECK:             hlfir.destroy %[[VAL_66]] : !hlfir.expr<!fir.char<1,2>>
! CHECK-NOT:         hlfir.destroy %[[VAL_83]]
! CHECK:             hlfir.end_associate %[[VAL_51]]#1, %[[VAL_51]]#2 : !fir.ref<!fir.char<1>>, i1
! CHECK-NOT:         hlfir.destroy %[[VAL_83]]
! CHECK:             hlfir.destroy %[[VAL_48]] : !hlfir.expr<!fir.char<1>>
! CHECK-NOT:         hlfir.destroy %[[VAL_83]]
! CHECK:             hlfir.yield_element %[[VAL_83]] : !hlfir.expr<!fir.char<1,4>>
! CHECK-NOT:         hlfir.destroy %[[VAL_83]]
! CHECK:           }
! CHECK-NOT:       hlfir.destroy %[[VAL_83]]
