! Test lowering of ANY intrinsic to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s
! simple 1 argument ANY
subroutine any1(a, s)
  logical :: a(:), s
  s = ANY(a)
end subroutine
! CHECK-LABEL: func.func @_QPany1(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:           %[[ARG1:.*]]: !fir.ref<!fir.logical<4>>
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.any %[[MASK]]#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0  : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! any with by-ref DIM argument
subroutine any2(a, s, d)
  logical :: a(:,:), s(:)
  integer :: d
  s = ANY(a, d)
end subroutine
! CHECK-LABEL: func.func @_QPany2(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "a"}
! CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "s"}
! CHECK:           %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[DIM_REF:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-NEXT:    %[[DIM:.*]] = fir.load %[[DIM_REF]]#0 : !fir.ref<i32>
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.any %[[MASK]]#0 dim %[[DIM]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?x!fir.logical<4>>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<?x!fir.logical<4>>, !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! any with DIM argument by-val, mask isn't boxed
subroutine any3(s)
  logical :: s(2)
  logical :: a(2,2) = reshape((/.true.,.false.,.true.,.false./), shape(a))
  s = ANY(a, 1)
end subroutine
! CHECK-LABEL: func.func @_QPany3(
! CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.array<2x!fir.logical<4>>> {fir.bindc_name = "s"}
! CHECK-DAG:     %[[ADDR:.*]] = fir.address_of{{.*}} : !fir.ref<!fir.array<2x2x!fir.logical<4>>>
! CHECK-DAG:     %[[MASK_SHAPE:.*]] = fir.shape {{.*}} -> !fir.shape<2>
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ADDR]](%[[MASK_SHAPE]])
! CHECK-DAG:     %[[OUT_SHAPE:.*]] = fir.shape {{.*}} -> !fir.shape<1>
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG0]](%[[OUT_SHAPE]])
! CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
! CHECK-DAG:     %[[EXPR:.*]] = hlfir.any %[[MASK]]#0 dim %[[C1]] : (!fir.ref<!fir.array<2x2x!fir.logical<4>>>, i32) -> !hlfir.expr<2x!fir.logical<4>>
! CHECK-DAG:     hlfir.assign %[[EXPR]] to %[[OUT]]
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<2x!fir.logical<4>>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! any with DIM from pointer
subroutine any4(a, s, d)
  integer, pointer :: d
  logical :: a(:,:), s(:)
  s = ANY(a, (d))
end subroutine
! CHECK-LABEL: func.func @_QPany4(
! CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "a"}
! CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "s"}
! CHECK:           %[[ARG2:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "d"}
! CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-DAG:     %[[DIM:.*]]:2 = hlfir.declare %[[ARG2]]
! CHECK-NEXT:    %[[DIM_BOX:.*]] = fir.load %[[DIM]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK-NEXT:    %[[DIM_ADDR:.*]] = fir.box_addr %[[DIM_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK-NEXT:    %[[DIM0:.*]] = fir.load %[[DIM_ADDR]] : !fir.ptr<i32>
! CHECK-NEXT:    %[[DIM1:.*]] = hlfir.no_reassoc %[[DIM0]] : i32
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.any %[[ARRAY]]#0 dim %[[DIM1]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?x!fir.logical<4>>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0 : !hlfir.expr<?x!fir.logical<4>>, !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK-NEXT:    hlfir.destroy %[[EXPR]] : !hlfir.expr<?x!fir.logical<4>>
! CHECK-NEXT:    return
! CHECK-NEXT:  }

subroutine any5(a, s)
  logical, allocatable :: a(:)
  logical :: s
  s = ANY(a)
end subroutine
! CHECK-LABEL: func.func @_QPany5(
! CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:           %[[ARG1:.*]]: !fir.ref<!fir.logical<4>>
! CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK-DAG:     %[[OUT:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK-NEXT:    %[[MASK_LOADED:.*]] = fir.load %[[MASK]]#0
! CHECK-NEXT:    %[[EXPR:.*]] = hlfir.any %[[MASK_LOADED]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.logical<4>
! CHECK-NEXT:    hlfir.assign %[[EXPR]] to %[[OUT]]#0  : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK-NEXT:    return
! CHECK-NEXT:  }
