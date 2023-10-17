! Test lowering of COUNT intrinsic to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

! simple 1 argument COUNT
subroutine count1(a, s)
  logical :: a(:)
  integer :: s
  s = COUNT(a)
end subroutine
! CHECK-LABEL: func.func @_QPcount1(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:           %[[ARG1:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.count %[[MASK]]#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : i32, !fir.ref<i32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! count with by-ref DIM argument
subroutine count2(a, s, d)
  logical :: a(:,:)
  integer :: s(:), d
  s = COUNT(a, d)
end subroutine
! CHECK-LABEL: func.func @_QPcount2(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>
! CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>
! CHECK:           %[[ARG2:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[DIM_REF:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[DIM:.*]] = fir.load %[[DIM_REF]]#0 : !fir.ref<i32>
! CHECK-DAG:     %[[EXPR:.*]] = hlfir.count %[[MASK]]#0 dim %[[DIM]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?xi32>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<?xi32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! count with DIM argument by-val, mask isn't boxed
subroutine count3(s)
  integer :: s(2)
  logical :: a(2,2) = reshape((/.true.,.false.,.true.,.false./), shape(a))
  s = COUNT(a, 1)
end subroutine
! CHECK-LABEL: func.func @_QPcount3(
! CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.array<2xi32>>
! CHECK-DAG:     %[[ADDR:.*]] = fir.address_of{{.*}} : !fir.ref<!fir.array<2x2x!fir.logical<4>>>
! CHECK-DAG:     %[[MASK_SHAPE:.*]] = fir.shape {{.*}} -> !fir.shape<2>
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ADDR]](%[[MASK_SHAPE]])
! CHECK-DAG:     %[[OUT_SHAPE:.*]] = fir.shape {{.*}} -> !fir.shape<1>
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG0]](%[[OUT_SHAPE]])
! CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
! CHECK-DAG:     %[[EXPR:.*]] = hlfir.count %[[MASK]]#0 dim %[[C1]] : (!fir.ref<!fir.array<2x2x!fir.logical<4>>>, i32) -> !hlfir.expr<2xi32>
! CHECK-DAG:     hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<2xi32>, !fir.ref<!fir.array<2xi32>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<2xi32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! count with dim and kind arguments
subroutine count4(a, s, d)
  logical :: a(:,:)
  integer :: s(:), d
  s = COUNT(a, d, 8)
end subroutine
! CHECK-LABEL: func.func @_QPcount4(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>
! CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>
! CHECK:           %[[ARG2:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[DIM_REF:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : i32
! CHECK-DAG:     %[[DIM:.*]] = fir.load %[[DIM_REF]]#0 : !fir.ref<i32>
! CHECK-DAG:     %[[EXPR:.*]] = hlfir.count %[[MASK]]#0 dim %[[DIM]] kind %[[C8]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32, i32) -> !hlfir.expr<?xi64>
! CHECK-DAG:     %[[RES_SHAPE:.*]] = hlfir.shape_of %[[EXPR]]
! CHECK-DAG:     %[[RES:.*]] = hlfir.elemental %[[RES_SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32>
! CHECK-DAG:     hlfir.assign %[[RES]] to %[[OUT]]#0
! CHECK-NEXT:    hlfir.destroy %[[RES]] : !hlfir.expr<?xi32>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<?xi64>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

subroutine count5(a, s)
  logical, allocatable :: a(:)
  integer :: s
  s = COUNT(a)
end subroutine
! CHECK-LABEL: func.func @_QPcount5(
! CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:           %[[ARG1:.*]]: !fir.ref<i32>
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-NEXT:    %[[MASK_LOADED:.*]] = fir.load %[[MASK]]#0
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.count %[[MASK_LOADED]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> i32
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : i32, !fir.ref<i32>
! CHECK-NEXT:    return
! CHECK-NEXT:  }
