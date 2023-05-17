! Test lowering of SUM intrinsic to HLFIR
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

! simple 1 argument SUM
subroutine sum1(a, s)
  integer :: a(:), s
  s = SUM(a)
end subroutine
! CHECK-LABEL: func.func @_QPsum1(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.sum %[[ARRAY]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>) ->
!hlfir.expr<i32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<i32>, !fir.ref<i32>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! sum with by-ref DIM argument
subroutine sum2(a, s, d)
  integer :: a(:,:), s(:), d
  s = SUM(a, d)
end subroutine
! CHECK-LABEL: func.func @_QPsum2(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[DIM_REF:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[DIM:.*]] = fir.load %[[DIM_REF]]#0 : !fir.ref<i32>
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.sum %[[ARRAY]]#0 dim %[[DIM]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! sum with scalar mask argument
subroutine sum3(a, s, m)
  integer :: a(:), s
  logical :: m
  s = SUM(a, m)
end subroutine
! CHECK-LABEL: func.func @_QPsum3(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.ref<!fir.logical<4>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.sum %[[ARRAY]]#0 mask %[[MASK]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.logical<4>>) -> !hlfir.expr<i32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<i32>, !fir.ref<i32>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! sum with array mask argument
subroutine sum4(a, s, m)
  integer :: a(:), s
  logical :: m(:)
  s = SUM(a, m)
end subroutine
! CHECK-LABEL: func.func @_QPsum4(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}, %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.sum %[[ARRAY]]#0 mask %[[MASK]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> !hlfir.expr<i32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<i32>, !fir.ref<i32>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! sum with all 3 arguments, dim is by-val, array isn't boxed
subroutine sum5(s)
  integer :: s(2)
  integer :: a(2,2) = reshape((/1, 2, 3, 4/), [2,2])
  s = sum(a, 1, .true.)
end subroutine
! CHECK-LABEL: func.func @_QPsum5
! CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.array<2xi32>>
! CHECK-DAG:     %[[ADDR:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.array<2x2xi32>>
! CHECK-DAG:     %[[ARRAY_SHAPE:.*]] = fir.shape {{.*}} -> !fir.shape<2>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ADDR]](%[[ARRAY_SHAPE]])
! CHECK-DAG:     %[[OUT_SHAPE:.*]] = fir.shape {{.*}} -> !fir.shape<1>
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG0]](%[[OUT_SHAPE]])
! CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
! CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.sum %[[ARRAY]]#0 dim %[[C1]] mask %[[TRUE]] {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<2x2xi32>>, i32, i1) -> !hlfir.expr<2xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<2xi32>, !fir.ref<!fir.array<2xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<2xi32>
! CHECK-NEXT:    return
! CHECK-nEXT:  }

subroutine sum6(a, s, d)
  integer, pointer :: d
  real :: a(:,:), s(:)
  s = sum(a, (d))
end subroutine
! CHECK-LABEL: func.func @_QPsum6(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>>
! CHECK:           %[[ARG2:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[DIM_VAR:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:     %[[DIM_BOX:.*]] = fir.load %[[DIM_VAR]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK-NEXT:    %[[DIM_ADDR:.*]] = fir.box_addr %[[DIM_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK-NEXT:    %[[DIM0:.*]] = fir.load %[[DIM_ADDR]] : !fir.ptr<i32>
! CHECK-NEXT:    %[[DIM1:.*]] = hlfir.no_reassoc %[[DIM0]] : i32
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.sum %[[ARRAY]]#0 dim %[[DIM1]] {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xf32>>, i32) -> !hlfir.expr<?xf32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }
