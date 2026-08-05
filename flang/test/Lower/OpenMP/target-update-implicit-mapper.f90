! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | \
! RUN:   FileCheck %s

subroutine implicit_mapper_to()
  type :: t
    integer, allocatable :: a(:)
  end type
  type(t) :: x

  !$omp target update to(x)
end subroutine

subroutine implicit_mapper_from()
  type :: inner
    integer, allocatable :: a(:)
  end type
  type :: outer
    type(inner) :: nested
  end type
  type(outer) :: x

  !$omp target update from(x)
end subroutine

subroutine implicit_mapper_iterator()
  type :: t
    integer, allocatable :: a(:)
  end type
  type(t) :: x(4)
  integer :: i

  !$omp target update to(iterator(i = 1:4): x(i))
end subroutine

! CHECK-DAG: omp.declare_mapper
! CHECK-DAG: omp.declare_mapper
! CHECK-DAG: omp.declare_mapper

! CHECK-LABEL: func.func @_QPimplicit_mapper_to
! CHECK: omp.map.info {{.*}} map_clauses(to) {{.*}}mapper(@{{.*}})
! CHECK: omp.target_update

! CHECK-LABEL: func.func @_QPimplicit_mapper_from
! CHECK: omp.map.info {{.*}} map_clauses(from) {{.*}}mapper(@{{.*}})
! CHECK: omp.target_update

! CHECK-LABEL: func.func @_QPimplicit_mapper_iterator
! CHECK: omp.iterator
! CHECK: omp.map.info {{.*}} map_clauses(to) {{.*}}mapper(@{{.*}})
! CHECK: omp.target_update {{.*}}map_iterated
