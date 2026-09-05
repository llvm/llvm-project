! Regression test for https://github.com/llvm/llvm-project/issues/218760
! The DO CONCURRENT -> OpenMP device conversion used to abort on arrays
! whose element type contains a nested derived-type component. Verify that
! a nested derived type with no allocatable members does not require an
! implicit mapper and converts successfully, while a nested derived type
! with allocatable members properly generates implicit mappers (including
! when the nested component is an array of records).

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

module nested_alloc_mod
  implicit none
  type :: inner_alloc_t
    real, allocatable :: values(:)
  end type

  type :: outer_alloc_t
    type(inner_alloc_t) :: inner(2)
  end type
end module nested_alloc_mod

! CHECK: omp.declare_mapper @[[INNER_MAPPER:.*inner_alloc_t.*]] : !fir.type<{{.*}}inner_alloc_t{{.*}}>
! CHECK: omp.declare_mapper @[[OUTER_MAPPER:.*outer_alloc_t.*]] : !fir.type<{{.*}}outer_alloc_t{{.*}}> {
! CHECK:   omp.map.info {{.*}} mapper(@[[INNER_MAPPER]])

subroutine nested_derived()
  implicit none

  type :: inner_t
    integer :: x
  end type

  type :: outer_t
    type(inner_t) :: member
  end type

  type(outer_t) :: a(4)
  integer :: i

  do concurrent (i = 1:4)
    a(i)%member%x = i
  end do
end subroutine

! CHECK-LABEL: func.func @{{.*}}nested_derived()
! CHECK:   %[[ARR_A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "{{.*}}a"}
! CHECK-NOT: mapper(
! CHECK:   omp.map.info var_ptr(%[[ARR_A]]#1 : {{.*}}) map_clauses(implicit, tofrom) capture(ByRef) {{.*}} name("{{.*}}a")
! CHECK:   omp.target
! CHECK:   omp.teams
! CHECK:   omp.parallel
! CHECK:   omp.distribute
! CHECK:   omp.wsloop
! CHECK:   omp.loop_nest

subroutine nested_derived_alloc()
  use nested_alloc_mod
  implicit none

  type(outer_alloc_t) :: a(4)
  integer :: i

  do concurrent (i = 1:4)
    a(1)%inner(1)%values = [1.0, 2.0]
  end do
end subroutine

! CHECK-LABEL: func.func @{{.*}}nested_derived_alloc()
! CHECK:   %[[ARR_ALLOC:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "{{.*}}a"}
! CHECK:   omp.map.info var_ptr(%[[ARR_ALLOC]]#1 : {{.*}}) map_clauses(implicit, tofrom) capture(ByRef) mapper(@[[OUTER_MAPPER]]) {{.*}} name("{{.*}}a")
