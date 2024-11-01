! Test lowering of MATMUL intrinsic to HLFIR
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

subroutine matmul1(lhs, rhs, res)
  integer :: lhs(:,:), rhs(:,:), res(:,:)
  res = MATMUL(lhs, rhs)
endsubroutine
! CHECK-LABEL: func.func @_QPmatmul1
! CHECK:           %[[LHS:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "lhs"}
! CHECK:           %[[RHS:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "rhs"}
! CHECK:           %[[RES:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "res"}
! CHECK-DAG:     %[[LHS_VAR:.*]]:2 = hlfir.declare %[[LHS]]
! CHECK-DAG:     %[[RHS_VAR:.*]]:2 = hlfir.declare %[[RHS]]
! CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[RES]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.matmul %[[LHS_VAR]]#0 %[[RHS_VAR]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>) -> !hlfir.expr<?x?xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[RES_VAR]]#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:   }
