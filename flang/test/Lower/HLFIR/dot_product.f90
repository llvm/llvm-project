! Test lowering of DOT_PRODUCT intrinsic to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

! dot product with numerical arguments
subroutine dot_product1(lhs, rhs, res)
  integer lhs(:), rhs(:), res
  res = DOT_PRODUCT(lhs,rhs)
end subroutine
! CHECK-LABEL: func.func @_QPdot_product1
! CHECK:           %[[LHS:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "lhs"}
! CHECK:           %[[RHS:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "rhs"}
! CHECK:           %[[RES:.*]]: !fir.ref<i32> {fir.bindc_name = "res"}
! CHECK-DAG:     %[[LHS_VAR:.*]]:2 = hlfir.declare %[[LHS]]
! CHECK-DAG:     %[[RHS_VAR:.*]]:2 = hlfir.declare %[[RHS]]
! CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[RES]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.dot_product %[[LHS_VAR]]#0 %[[RHS_VAR]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>) -> i32
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[RES_VAR]]#0 : i32, !fir.ref<i32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! dot product with logical arguments
subroutine dot_product2(lhs, rhs, res)
  logical lhs(:), rhs(:), res
  res = DOT_PRODUCT(lhs,rhs)
end subroutine
! CHECK-LABEL: func.func @_QPdot_product2
! CHECK:           %[[LHS:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "lhs"}
! CHECK:           %[[RHS:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "rhs"}
! CHECK:           %[[RES:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "res"}
! CHECK-DAG:     %[[LHS_VAR:.*]]:2 = hlfir.declare %[[LHS]]
! CHECK-DAG:     %[[RHS_VAR:.*]]:2 = hlfir.declare %[[RHS]]
! CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[RES]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.dot_product %[[LHS_VAR]]#0 %[[RHS_VAR]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[RES_VAR]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! arguments are of known shape
subroutine dot_product3(lhs, rhs, res)
  integer lhs(5), rhs(5), res
  res = DOT_PRODUCT(lhs,rhs)
end subroutine
! CHECK-LABEL: func.func @_QPdot_product3
! CHECK:           %[[LHS:.*]]: !fir.ref<!fir.array<5xi32>> {fir.bindc_name = "lhs"}
! CHECK:           %[[RHS:.*]]: !fir.ref<!fir.array<5xi32>> {fir.bindc_name = "rhs"}
! CHECK:           %[[RES:.*]]: !fir.ref<i32> {fir.bindc_name = "res"}
! CHECK-DAG:     %[[LHS_VAR:.*]]:2 = hlfir.declare %[[LHS]]
! CHECK-DAG:     %[[RHS_VAR:.*]]:2 = hlfir.declare %[[RHS]]
! CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[RES]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.dot_product %[[LHS_VAR]]#0 %[[RHS_VAR]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>) -> i32
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[RES_VAR]]#0 : i32, !fir.ref<i32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }
