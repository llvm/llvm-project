! Test lowering of TRANSPOSE intrinsic to HLFIR
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

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
