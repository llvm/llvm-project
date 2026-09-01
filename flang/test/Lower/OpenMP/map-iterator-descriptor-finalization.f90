! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | \
! RUN:   FileCheck %s --implicit-check-not=fir.box_offset \
! RUN:   --implicit-check-not=map_entries \
! RUN:   --implicit-check-not='map_clauses(attach'

! Verify that descriptor-backed iterator maps describe only selected data and
! MapInfoFinalization does not add descriptor parent or attachment maps.

subroutine iter_map_assumed_shape(a, n)
  integer :: a(:)
  integer :: n
  integer :: i

  !$omp target enter data map(iterator(i = 1:n), to: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPiter_map_assumed_shape(
! CHECK: %[[IT:.*]] = omp.iterator
! CHECK: %[[ADDR:.*]] = fir.box_addr
! CHECK: %[[MAP:.*]] = omp.map.info
! CHECK-SAME: var_ptr(%[[ADDR]]
! CHECK-SAME: map_clauses(to) capture(ByRef)
! CHECK-SAME: bounds(%{{.*}})
! CHECK: omp.yield(%[[MAP]]
! CHECK: } -> !omp.iterated
! CHECK: omp.target_enter_data map_iterated(%[[IT]]

subroutine iter_map_allocatable(a, n)
  integer, allocatable :: a(:)
  integer :: n
  integer :: i

  !$omp target enter data map(iterator(i = 1:n), to: a(i))
end subroutine

! CHECK-LABEL: func.func @_QPiter_map_allocatable(
! CHECK: %[[IT:.*]] = omp.iterator
! CHECK: %[[BOX:.*]] = fir.load
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]]
! CHECK: %[[MAP:.*]] = omp.map.info
! CHECK-SAME: var_ptr(%[[ADDR]]
! CHECK-SAME: map_clauses(to) capture(ByRef)
! CHECK-SAME: bounds(%{{.*}})
! CHECK: omp.yield(%[[MAP]]
! CHECK: } -> !omp.iterated
! CHECK: omp.target_enter_data map_iterated(%[[IT]]
