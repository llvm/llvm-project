! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

program reduce
  integer a(0:1)

  call sub(a, 0, 1)

contains
  subroutine sub(a, lb, ub)
    integer :: i, lb, ub, a(lb:ub)

    !$omp parallel do reduction(+:a)
      do i = 1, 10
        a(0) = a(0) + 1
      end do
    !$omp end parallel do
  end subroutine

end program

! CHECK-LABEL:   omp.declare_reduction @add_reduction_byref_box_Uxi32 : !fir.ref<!fir.box<!fir.array<?xi32>>> alloc {
! CHECK:         } combiner {
! CHECK:         ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.box<!fir.array<?xi32>>>, %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.array<?xi32>>>):
! CHECK:           %[[ARR0:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           %[[ARR1:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[DIMS:.*]]:3 = fir.box_dims %[[ARR0]], %[[C0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[C1]], %[[DIMS]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[C1_0:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[ARG2:.*]] = %[[C1_0]] to %[[DIMS]]#1 step %[[C1_0]] unordered {
! CHECK:             %[[COOR0:.*]] = fir.array_coor %[[ARR0]](%[[SHAPE_SHIFT]]) %[[ARG2]] : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[COOR1:.*]] = fir.array_coor %[[ARR1]](%[[SHAPE_SHIFT]]) %[[ARG2]] : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:             %[[ELEM0:.*]] = fir.load %[[COOR0]] : !fir.ref<i32>
! CHECK:             %[[ELEM1:.*]] = fir.load %[[COOR1]] : !fir.ref<i32>
! CHECK:             %[[SUM:.*]] = arith.addi %[[ELEM0]], %[[ELEM1]] : i32
! CHECK:             fir.store %[[SUM]] to %[[COOR0]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           omp.yield(%[[ARG0]] : !fir.ref<!fir.box<!fir.array<?xi32>>>)
! CHECK:         }

! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "reduce"} {
! CHECK:           omp.wsloop {{.*}} reduction(byref @add_reduction_byref_box_Uxi32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>)
