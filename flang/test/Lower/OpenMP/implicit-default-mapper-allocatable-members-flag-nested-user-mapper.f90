! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -fno-implicit-default-mapper-allocatable-members %s -o - | FileCheck %s

! Check that -fno-implicit-default-mapper-allocatable-members does not suppress
! generation of an implicit default mapper when a nested record component has a
! user-defined default mapper. The generated outer mapper is still needed as a
! carrier for the nested component mapper, but it must not emit maps for
! allocatable components.

program main
  implicit none

  type :: inner(k)
    integer, kind :: k
    integer(k), pointer :: p(:)
  end type

  !$omp declare mapper(default: inner(4) :: v) map(tofrom: v%p)

  type :: outer
    integer, allocatable :: unused(:)
    type(inner(4)) :: child
  end type

contains
  subroutine run(x)
    type(outer) :: x
    !$omp target map(tofrom: x)
      x%child%p(1) = 42
    !$omp end target
  end subroutine
end program

! CHECK-DAG: omp.declare_mapper @[[OUTER_MAPPER:.*outer.*_omp_default_mapper]]
! CHECK-DAG: omp.declare_mapper @[[INNER_MAPPER:.*inner.*_omp_default_mapper]]
! CHECK-DAG: fir.coordinate_of %{{.*}}, child
! CHECK-DAG: omp.map.info {{.*}}mapper(@[[INNER_MAPPER]])
! CHECK-DAG: omp.map.info {{.*}}members(%{{.*}} : [1]
! CHECK: omp.map.info {{.*}}mapper(@[[OUTER_MAPPER]]) name("x")
! CHECK: omp.target
