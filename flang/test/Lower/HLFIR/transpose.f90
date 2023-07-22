! Test lowering of TRANSPOSE intrinsic to HLFIR
! RUN: bbc -emit-hlfir --polymorphic-type -o - %s 2>&1 | FileCheck %s

subroutine transpose1(m, res)
  integer :: m(1,2), res(2, 1)
  res = TRANSPOSE(m)
endsubroutine
! CHECK-LABEL: func.func @_QPtranspose1
! CHECK:           %[[M_ARG:.*]]: !fir.ref<!fir.array<1x2xi32>>
! CHECK:           %[[RES_ARG:.*]]: !fir.ref<!fir.array<2x1xi32>>
! CHECK-DAG:     %[[ARG:.*]]:2 = hlfir.declare %[[M_ARG]](%[[M_SHAPE:.*]]) {[[NAME:.*]]} : (!fir.ref<!fir.array<1x2xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1x2xi32>>, !fir.ref<!fir.array<1x2xi32>>)
! CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[RES_ARG]](%[[RES_SHAPE:.*]]) {[[NAME2:.*]]} : (!fir.ref<!fir.array<2x1xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<2x1xi32>>, !fir.ref<!fir.array<2x1xi32>>)
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
! CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[RES_ARG]](%[[RES_SHAPE:.*]]) {[[NAME2:.*]]} : (!fir.ref<!fir.array<2x1xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<2x1xi32>>, !fir.ref<!fir.array<2x1xi32>>)
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
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_polymorphic_resultEm"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_polymorphic_resultEres"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
! CHECK:           %[[VAL_5:.*]] = hlfir.transpose %[[VAL_4]] : (!fir.class<!fir.heap<!fir.array<?x?xnone>>>) -> !hlfir.expr<?x?xnone?>
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_3]]#0 realloc : !hlfir.expr<?x?xnone?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>
! CHECK:           hlfir.destroy %[[VAL_5]] : !hlfir.expr<?x?xnone?>
! CHECK:           return
! CHECK:         }
