! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -fno-implicit-default-mapper-allocatable-members %s -o - | FileCheck %s

! Check that an implicit default mapper is generated for an outer record even
! when it has no allocatable components, if a nested record component has a
! user-defined default mapper. The generated outer mapper is needed as a carrier
! to attach the nested component mapper.

program main
  implicit none

  type :: inner(k)
    integer, kind :: k
    integer(k), pointer :: p(:)
  end type

  !$omp declare mapper(default: inner(4) :: v) map(tofrom: v%p)

  type :: outer
    integer :: some_arr(10)
    type(inner(4)) :: child
  end type

  type(outer) :: x

  !$omp target map(tofrom: x)
    x%child%p(1) = 42
  !$omp end target
end program

! CHECK-DAG: omp.declare_mapper @[[OUTER_MAPPER:.*outer.*_omp_default_mapper]]
! CHECK-DAG: omp.declare_mapper @[[INNER_MAPPER:.*inner.*_omp_default_mapper]]
! CHECK-DAG: fir.coordinate_of %{{.*}}, child
! CHECK-DAG: omp.map.info {{.*}}mapper(@[[INNER_MAPPER]])
! CHECK-DAG: omp.map.info {{.*}}members(%{{.*}} : [1]
! CHECK: omp.map.info {{.*}}mapper(@[[OUTER_MAPPER]]) name("x")
! CHECK: omp.target
