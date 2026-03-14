! Test lowering of MATMUL intrinsic to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

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

! regression test for a case where the AST and FIR have different amounts of
! shape inference
subroutine matmul2(c)
  integer, parameter :: N = 4
  integer, dimension(:,:), allocatable :: a, b, c
  integer, dimension(N,N) :: x

  allocate(a(3*N, N), b(N, N), c(3*N, N))

  call fill(a)
  call fill(b)
  call fill(x)

  c = matmul(a, b - x)
endsubroutine
! CHECK-LABEL: func.func @_QPmatmul2
! CHECK:           %[[C_ARG:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK:         %[[B_BOX_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>> {bindc_name = "b"
! CHECK:         %[[B_BOX_DECL:.*]]:2 = hlfir.declare %[[B_BOX_ALLOC]] {{.*}} uniq_name = "_QFmatmul2Eb"


! CHECK:         fir.call @_QPfill
! CHECK:         fir.call @_QPfill
! CHECK:         fir.call @_QPfill
! CHECK-NEXT:    %[[B_BOX:.*]] = fir.load %[[B_BOX_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK-NEXT:    %[[C0:.*]] = arith.constant 0 : index
! CHECK-NEXT:    %[[B_DIMS_0:.*]]:3 = fir.box_dims %[[B_BOX]], %[[C0]]
! CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
! CHECK-NEXT:    %[[B_DIMS_1:.*]]:3 = fir.box_dims %[[B_BOX]], %[[C1]]
! CHECK-NEXT:    %[[B_SHAPE:.*]] = fir.shape %[[B_DIMS_0]]#1, %[[B_DIMS_1]]#1
! CHECK-NEXT:    %[[ELEMENTAL:.*]] = hlfir.elemental %[[B_SHAPE]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {

! CHECK:         }
! CHECK-NEXT:    %[[A_BOX:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>

! The shapes in these types are what is being tested:
! CHECK-NEXT:    %[[MATMUL:.*]] = hlfir.matmul %[[A_BOX]] %[[ELEMENTAL]] {{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>, !hlfir.expr<?x?xi32>) -> !hlfir.expr<?x4xi32>

subroutine matmul3(lhs, rhs, res)
  integer, allocatable :: lhs(:,:), rhs(:,:), res(:,:)
  res = MATMUL(lhs, rhs)
endsubroutine
! CHECK-LABEL: func.func @_QPmatmul3
! CHECK:           %[[LHS:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>> {fir.bindc_name = "lhs"}
! CHECK:           %[[RHS:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>> {fir.bindc_name = "rhs"}
! CHECK:           %[[RES:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>> {fir.bindc_name = "res"}
! CHECK-DAG:     %[[LHS_VAR:.*]]:2 = hlfir.declare %[[LHS]]
! CHECK-DAG:     %[[RHS_VAR:.*]]:2 = hlfir.declare %[[RHS]]
! CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[RES]]
! CHECK-NEXT:    %[[LHS_LD:.*]] = fir.load %[[LHS_VAR]]#0
! CHECK-NEXT:    %[[RHS_LD:.*]] = fir.load %[[RHS_VAR]]#0
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.matmul %[[LHS_LD]] %[[RHS_LD]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>, !fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !hlfir.expr<?x?xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[RES_VAR]]#0 realloc : !hlfir.expr<?x?xi32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:   }
