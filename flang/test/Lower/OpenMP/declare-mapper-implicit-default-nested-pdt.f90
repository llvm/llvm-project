! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
!
! This file groups related OpenMP mapper-name regressions:
! 1) nested default mapper reuse for pointer components
! 2) distinct implicit mapper names for PDT kind instantiations
! 3) nested PDT reuse of an explicit declare mapper(default:)

module ptr_case
  type :: newdata
    real(8), pointer :: value
  end type
  !$omp declare mapper(default: newdata :: v) map(tofrom: v%value)

  type :: newvec
    integer :: len
    type(newdata), allocatable :: data(:)
  end type
contains
  subroutine trigger
    type(newvec) :: s
    !$omp target data map(s)
      s%len = 0
    !$omp end target data
  end subroutine
end module

! CHECK-DAG: omp.declare_mapper @{{.*}}newdata_omp_default_mapper
! CHECK-DAG: map_clauses(tofrom)
! CHECK-DAG: {name = "v%value"}
! CHECK-DAG: omp.declare_mapper @{{.*}}newvec_omp_default_mapper
! CHECK-DAG: mapper(@{{.*}}newdata_omp_default_mapper)

module kinds_case
  type :: pdt(k)
    integer, kind :: k
    real(k) :: x
  end type

  type :: holder4_kinds
    type(pdt(4)), allocatable :: a(:)
  end type

  type :: holder8_kinds
    type(pdt(8)), allocatable :: a(:)
  end type
contains
  subroutine trigger
    type(holder4_kinds) :: h4
    type(holder8_kinds) :: h8
    !$omp target data map(h4, h8)
    !$omp end target data
  end subroutine
end module

! CHECK-DAG: omp.declare_mapper @{{.*}}pdtK4_omp_default_mapper
! CHECK-DAG: omp.declare_mapper @{{.*}}pdtK8_omp_default_mapper
! CHECK-DAG: mapper(@{{.*}}pdtK4_omp_default_mapper)
! CHECK-DAG: mapper(@{{.*}}pdtK8_omp_default_mapper)

module reuse_case
  type :: pdt(k)
    integer, kind :: k
    real(k) :: x
  end type

  !$omp declare mapper(default: pdt(4) :: v) map(tofrom: v%x)

  type :: holder4_reuse
    type(pdt(4)), allocatable :: a(:)
  end type

  type :: holder8_reuse
    type(pdt(8)), allocatable :: a(:)
  end type
contains
  subroutine trigger
    type(holder4_reuse) :: h4
    type(holder8_reuse) :: h8
    !$omp target data map(h4, h8)
    !$omp end target data
  end subroutine
end module

! CHECK-DAG: omp.declare_mapper @{{.*}}pdt_omp_default_mapper : !fir.type<{{.*}}TpdtK4
! CHECK-DAG: mapper(@{{.*}}pdt_omp_default_mapper)
! CHECK-DAG: mapper(@{{.*}}pdtK8_omp_default_mapper)
