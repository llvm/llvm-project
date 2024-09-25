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

subroutine dot_product4(lhs, rhs, res)
  integer, allocatable :: lhs(:), rhs(:)
  integer :: res
  res = dot_product(lhs, rhs)
endsubroutine
! CHECK-LABEL: func.func @_QPdot_product4
! CHECK:           %[[LHS:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "lhs"}
! CHECK:           %[[RHS:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "rhs"}
! CHECK:           %[[RES:.*]]: !fir.ref<i32> {fir.bindc_name = "res"}
! CHECK-DAG:     %[[LHS_VAR:.*]]:2 = hlfir.declare %[[LHS]]
! CHECK-DAG:     %[[RHS_VAR:.*]]:2 = hlfir.declare %[[RHS]]
! CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[RES]]
! CHECK-NEXT:    %[[LHS_LD:.*]] = fir.load %[[LHS_VAR]]#0
! CHECK-NEXT:    %[[RHS_LD:.*]] = fir.load %[[RHS_VAR]]#0
! CHECK-NEXT:    %[[PROD:.*]] = hlfir.dot_product %[[LHS_LD]] %[[RHS_LD]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) -> i32
! CHECK-NEXT:    hlfir.assign %[[PROD]] to %[[RES_VAR]]#0 : i32, !fir.ref<i32>
! CHECK-NEXT:    return
! CHECK-NEXT:   }

! CHECK-LABEL: func.func @_QPdot_product5
! CHECK:    %[[LHS:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{[0-9]+}} {uniq_name = "_QFdot_product5Elhs"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:    %[[C3:.*]] = arith.constant 3 : index
! CHECK:    %[[RHS_SHAPE:.*]] = fir.shape %[[C3]] : (index) -> !fir.shape<1>
! CHECK:    %[[RHS:.*]]:2 = hlfir.declare %{{.*}}(%[[RHS_SHAPE]]) dummy_scope %{{[0-9]+}} {uniq_name = "_QFdot_product5Erhs"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
! CHECK:    {{.*}} = hlfir.dot_product %[[LHS]]#0 %[[RHS]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<3xi32>>) -> i32
subroutine dot_product5(lhs, rhs, res)
  integer :: lhs(:), rhs(3)
  integer :: res
  res = dot_product(lhs, rhs)
endsubroutine
