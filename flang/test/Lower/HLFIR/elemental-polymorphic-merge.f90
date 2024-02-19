! Test that the produced hlfir.elemental had proper result type and the mold.
! RUN: bbc --emit-hlfir --polymorphic-type -I nowhere -o - %s | FileCheck %s

subroutine test_polymorphic_merge(x, y, r, m)
  type t
  end type t
  class(t), allocatable :: r(:)
  class(t), intent(in) :: y(:), x
  logical :: m(:)
  r = merge(x, y, m)
end subroutine test_polymorphic_merge
! CHECK-LABEL:   func.func @_QPtest_polymorphic_merge(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.class<!fir.type<_QFtest_polymorphic_mergeTt>> {fir.bindc_name = "x"},
! CHECK-SAME:        %[[VAL_1:.*]]: !fir.class<!fir.array<?x!fir.type<_QFtest_polymorphic_mergeTt>>> {fir.bindc_name = "y"},
! CHECK-SAME:        %[[VAL_2:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFtest_polymorphic_mergeTt>>>>> {fir.bindc_name = "r"},
! CHECK-SAME:        %[[VAL_3:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "m"}) {
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFtest_polymorphic_mergeEm"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_polymorphic_mergeEr"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFtest_polymorphic_mergeTt>>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFtest_polymorphic_mergeTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFtest_polymorphic_mergeTt>>>>>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_polymorphic_mergeEx"} : (!fir.class<!fir.type<_QFtest_polymorphic_mergeTt>>) -> (!fir.class<!fir.type<_QFtest_polymorphic_mergeTt>>, !fir.class<!fir.type<_QFtest_polymorphic_mergeTt>>)
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_polymorphic_mergeEy"} : (!fir.class<!fir.array<?x!fir.type<_QFtest_polymorphic_mergeTt>>>) -> (!fir.class<!fir.array<?x!fir.type<_QFtest_polymorphic_mergeTt>>>, !fir.class<!fir.array<?x!fir.type<_QFtest_polymorphic_mergeTt>>>)
! CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_7]]#0, %[[VAL_8]] : (!fir.class<!fir.array<?x!fir.type<_QFtest_polymorphic_mergeTt>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_11:.*]] = hlfir.elemental %[[VAL_10]] mold %[[VAL_6]]#0 unordered : (!fir.shape<1>, !fir.class<!fir.type<_QFtest_polymorphic_mergeTt>>) -> !hlfir.expr<?x!fir.type<_QFtest_polymorphic_mergeTt>?> {
! CHECK:           ^bb0(%[[VAL_12:.*]]: index):
! CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_12]])  : (!fir.class<!fir.array<?x!fir.type<_QFtest_polymorphic_mergeTt>>>, index) -> !fir.class<!fir.type<_QFtest_polymorphic_mergeTt>>
! CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_12]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<!fir.logical<4>>
! CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.logical<4>) -> i1
! CHECK:             %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_6]]#1, %[[VAL_13]] : !fir.class<!fir.type<_QFtest_polymorphic_mergeTt>>
! CHECK:             %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_17]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.class<!fir.type<_QFtest_polymorphic_mergeTt>>) -> (!fir.class<!fir.type<_QFtest_polymorphic_mergeTt>>, !fir.class<!fir.type<_QFtest_polymorphic_mergeTt>>)
! CHECK:             %[[VAL_19:.*]] = hlfir.as_expr %[[VAL_18]]#0 : (!fir.class<!fir.type<_QFtest_polymorphic_mergeTt>>) -> !hlfir.expr<!fir.type<_QFtest_polymorphic_mergeTt>?>
! CHECK:             hlfir.yield_element %[[VAL_19]] : !hlfir.expr<!fir.type<_QFtest_polymorphic_mergeTt>?>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_11]] to %[[VAL_5]]#0 realloc : !hlfir.expr<?x!fir.type<_QFtest_polymorphic_mergeTt>?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFtest_polymorphic_mergeTt>>>>>
! CHECK:           hlfir.destroy %[[VAL_11]] : !hlfir.expr<?x!fir.type<_QFtest_polymorphic_mergeTt>?>
! CHECK:           return
! CHECK:         }
