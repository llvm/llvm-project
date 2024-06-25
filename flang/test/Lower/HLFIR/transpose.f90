! Test lowering of TRANSPOSE intrinsic to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine transpose1(m, res)
  integer :: m(1,2), res(2, 1)
  res = TRANSPOSE(m)
endsubroutine
! CHECK-LABEL: func.func @_QPtranspose1
! CHECK:           %[[M_ARG:.*]]: !fir.ref<!fir.array<1x2xi32>>
! CHECK:           %[[RES_ARG:.*]]: !fir.ref<!fir.array<2x1xi32>>
! CHECK-DAG:     %[[ARG:.*]]:2 = hlfir.declare %[[M_ARG]](%[[M_SHAPE:.*]]) dummy_scope %{{[0-9]+}} {[[NAME:.*]]} : (!fir.ref<!fir.array<1x2xi32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<1x2xi32>>, !fir.ref<!fir.array<1x2xi32>>)
! CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[RES_ARG]](%[[RES_SHAPE:.*]]) dummy_scope %{{[0-9]+}} {[[NAME2:.*]]} : (!fir.ref<!fir.array<2x1xi32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<2x1xi32>>, !fir.ref<!fir.array<2x1xi32>>)
! CHECK:         %[[EXPR:.*]] = hlfir.transpose %[[ARG]]#0 : (!fir.ref<!fir.array<1x2xi32>>) -> !hlfir.expr<2x1xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[RES]]#0
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! test the case where lowering has more exact information about the output
! shape than is available from the argument
subroutine transpose2(a, out)
  real, allocatable, dimension(:) :: a
  real, dimension(:,:) :: out
  integer, parameter :: N = 3
  integer, parameter :: M = 4

  allocate(a(N*M))
  out = transpose(reshape(a, (/N, M/)))
end subroutine
! CHECK-LABEL: func.func @_QPtranspose2(

subroutine transpose3(m, res)
  integer, allocatable :: m(:,:)
  integer :: res(2, 1)
  res = TRANSPOSE(m)
endsubroutine
! CHECK-LABEL: func.func @_QPtranspose3
! CHECK:           %[[M_ARG:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK:           %[[RES_ARG:.*]]: !fir.ref<!fir.array<2x1xi32>>
! CHECK-DAG:     %[[ARG:.*]]:2 = hlfir.declare %[[M_ARG]]
! CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[RES_ARG]](%[[RES_SHAPE:.*]]) dummy_scope %{{[0-9]+}} {[[NAME2:.*]]} : (!fir.ref<!fir.array<2x1xi32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<2x1xi32>>, !fir.ref<!fir.array<2x1xi32>>)
! CHECK:         %[[ARG_LOADED:.*]] = fir.load %[[ARG]]#0
! CHECK:         %[[EXPR:.*]] = hlfir.transpose %[[ARG_LOADED]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !hlfir.expr<?x?xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[RES]]#0
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! Test that the result type is polymorphic.
subroutine test_polymorphic_result(m, res)
  class(*), allocatable, dimension(:, :) :: m, res
  res = transpose(m)
end subroutine test_polymorphic_result
! CHECK-LABEL:   func.func @_QPtest_polymorphic_result(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>> {fir.bindc_name = "m"},
! CHECK-SAME:        %[[VAL_1:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>> {fir.bindc_name = "res"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_polymorphic_resultEm"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_polymorphic_resultEres"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
! CHECK:           %[[VAL_5:.*]] = hlfir.transpose %[[VAL_4]] : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>) -> !hlfir.expr<?x?xnone?>
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_3]]#0 realloc : !hlfir.expr<?x?xnone?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
! CHECK:           hlfir.destroy %[[VAL_5]] : !hlfir.expr<?x?xnone?>
! CHECK:           return
! CHECK:         }

! Test that hlfir.transpose lowering inherits constant
! character length from the argument, when the length
! is uknown from the Fortran::evaluate expression type.
subroutine test_unknown_char_len_result
  character(len=3), dimension(3,3) :: w
  character(len=2), dimension(3,3) :: w2
  w2 = transpose(w(:,:)(1:2))
end subroutine test_unknown_char_len_result
! CHECK-LABEL:   func.func @_QPtest_unknown_char_len_result() {
! CHECK:           %[[VAL_0:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_1:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.array<3x3x!fir.char<1,3>> {bindc_name = "w", uniq_name = "_QFtest_unknown_char_len_resultEw"}
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_4]]) typeparams %[[VAL_0]] {uniq_name = "_QFtest_unknown_char_len_resultEw"} : (!fir.ref<!fir.array<3x3x!fir.char<1,3>>>, !fir.shape<2>, index) -> (!fir.ref<!fir.array<3x3x!fir.char<1,3>>>, !fir.ref<!fir.array<3x3x!fir.char<1,3>>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_7:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_8:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.array<3x3x!fir.char<1,2>> {bindc_name = "w2", uniq_name = "_QFtest_unknown_char_len_resultEw2"}
! CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_10]]) typeparams %[[VAL_6]] {uniq_name = "_QFtest_unknown_char_len_resultEw2"} : (!fir.ref<!fir.array<3x3x!fir.char<1,2>>>, !fir.shape<2>, index) -> (!fir.ref<!fir.array<3x3x!fir.char<1,2>>>, !fir.ref<!fir.array<3x3x!fir.char<1,2>>>)
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_14:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_16:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_14]], %[[VAL_16]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_19:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_20:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_21:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_12]]:%[[VAL_1]]:%[[VAL_13]], %[[VAL_12]]:%[[VAL_2]]:%[[VAL_15]]) substr %[[VAL_18]], %[[VAL_19]]  shape %[[VAL_17]] typeparams %[[VAL_20]] : (!fir.ref<!fir.array<3x3x!fir.char<1,3>>>, index, index, index, index, index, index, index, index, !fir.shape<2>, index) -> !fir.box<!fir.array<3x3x!fir.char<1,2>>>
! CHECK:           %[[VAL_22:.*]] = hlfir.transpose %[[VAL_21]] : (!fir.box<!fir.array<3x3x!fir.char<1,2>>>) -> !hlfir.expr<3x3x!fir.char<1,2>>
! CHECK:           hlfir.assign %[[VAL_22]] to %[[VAL_11]]#0 : !hlfir.expr<3x3x!fir.char<1,2>>, !fir.ref<!fir.array<3x3x!fir.char<1,2>>>
! CHECK:           hlfir.destroy %[[VAL_22]] : !hlfir.expr<3x3x!fir.char<1,2>>
! CHECK:           return
! CHECK:         }
